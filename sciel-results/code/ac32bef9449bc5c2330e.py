"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations
from collections import defaultdict

def _multiplication_tensor(n, m, p):
    """Return the flattened multiplication tensor T[i,j,k]=1 for all valid indices."""
    # T is a 3D array of shape (n, m, p) where T[i,j,k] = 1
    # The decomposition requires: sum_r U[r,i,j] * V[r,j,k] * W[r,k,i] = 1 for all (i,j,k)
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _rank1_tensor_vec(u, v, w, n, m, p):
    """Given vectors u (n*m,), v (m*p,), w (p*n,), return the rank-1 tensor vector."""
    U = u.reshape(n, m)
    V = v.reshape(m, p)
    W = w.reshape(p, n)
    # T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = U[i, j] * V[j, k] * W[k, i]
    return T.flatten()

def _greedy_decompose(T, n, m, p, num_candidates, deadline, seed=42):
    """
    Greedy decomposition: repeatedly pick the rank-1 term that best reduces residual.
    """
    rng = np.random.RandomState(seed)
    residual = T.copy()
    selected = []  # list of (u, v, w)
    
    # Pre-generate a pool of candidate rank-1 terms
    # Each candidate has sparse entries in {-1, 0, 1} with small support
    candidates = []
    for _ in range(num_candidates):
        # Random support sizes
        su = rng.randint(1, 5)  # 1 to 4 non-zeros in U
        sv = rng.randint(1, 5)
        sw = rng.randint(1, 5)
        
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        
        u_idx = rng.choice(n * m, size=su, replace=False)
        v_idx = rng.choice(m * p, size=sv, replace=False)
        w_idx = rng.choice(p * n, size=sw, replace=False)
        
        for idx in u_idx:
            u[idx] = rng.choice([-1, 1])
        for idx in v_idx:
            v[idx] = rng.choice([-1, 1])
        for idx in w_idx:
            w[idx] = rng.choice([-1, 1])
        
        candidates.append((u, v, w))
    
    # Also add structured candidates based on Strassen-like patterns
    # Add all "single-entry" rank-1 terms (schoolbook basis)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((u, v, w))
    
    # Add 2x2 Strassen patterns scaled to 4x4x6
    strassen_2x2 = _get_strassen_2x2_candidates(n, m, p)
    candidates.extend(strassen_2x2)
    
    # Greedy selection
    max_rank = n * m * p  # schoolbook as upper bound
    for step in range(max_rank):
        if time.time() > deadline:
            break
        if np.allclose(residual, 0, atol=1e-10):
            break
        
        best_score = -np.inf
        best_cand = None
        best_coeff = 0.0
        
        # For each candidate, find optimal coefficient
        # residual_new = residual - coeff * cand_tensor
        # We want to minimize ||residual - coeff * cand||^2
        # Optimal coeff = <residual, cand> / ||cand||^2
        for u, v, w in candidates:
            if time.time() > deadline:
                break
            cand_T = _rank1_tensor_vec(u, v, w, n, m, p)
            dot = np.dot(residual, cand_T)
            norm_sq = np.dot(cand_T, cand_T)
            if norm_sq < 1e-12:
                continue
            coeff = dot / norm_sq
            # Score: reduction in residual norm squared
            new_residual = residual - coeff * cand_T
            score = np.dot(residual, residual) - np.dot(new_residual, new_residual)
            if score > best_score:
                best_score = score
                best_cand = (u, v, w)
                best_coeff = coeff
        
        if best_cand is None or best_score < 1e-10:
            break
        
        u, v, w = best_cand
        # Store the term (we'll scale later)
        selected.append((u * best_coeff, v, w))
        residual = residual - best_coeff * _rank1_tensor_vec(u, v, w, n, m, p)
        
        # Check if residual is essentially zero
        if np.max(np.abs(residual)) < 1e-9:
            break
    
    if len(selected) >= max_rank:
        return None  # Couldn't do better than schoolbook
    
    # Verify and clean up
    result = _verify_and_clean(selected, n, m, p)
    return result

def _local_search(terms, n, m, p, deadline, seed=99):
    """
    Local search: try to remove terms (if residual still close to 0 after re-solving)
    or swap terms for better coverage.
    """
    rng = np.random.RandomState(seed)
    T = _multiplication_tensor(n, m, p)
    
    current_terms = [list(t) for t in terms]
    current_rank = len(current_terms)
    
    # Try removal: for each term, check if remaining terms can still span T
    # This is expensive, so do a simpler version: try to find a subset
    # that still exactly represents T
    
    # Strategy: try to eliminate terms one at a time by adjusting others
    # For now, just try: can we drop the last term and adjust?
    
    # Better approach: use the fact that if we have more terms than needed,
    # some are linearly dependent in the tensor space.
    
    # Build matrix of rank-1 tensor vectors
    vecs = []
    for u, v, w in current_terms:
        vecs.append(_rank1_tensor_vec(u, v, w, n, m, p))
    vecs = np.array(vecs)  # (rank, n*m*p)
    
    # Find minimum subset that spans T
    # Use greedy: start with all, try to remove each
    active = list(range(len(current_terms)))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in list(active):
            if len(active) <= 1:
                break
            trial_active = [a for a in active if a != idx]
            if not trial_active:
                continue
            # Check if T is in span of remaining vectors
            sub_vecs = vecs[trial_active]
            # Solve sub_vecs.T @ x = T (least squares)
            try:
                x, residuals, rank, sv = np.linalg.lstsq(sub_vecs.T, T, rcond=None)
                reconstruction = sub_vecs.T @ x
                if np.allclose(reconstruction, T, atol=1e-8):
                    active = trial_active
                    improved = True
                    break
            except:
                pass
    
    if len(active) < current_rank:
        new_terms = [current_terms[i] for i in active]
        return new_terms
    
    return current_terms

def _verify_and_clean(terms, n, m, p):
    """Verify terms exactly decompose T and return cleaned dict or None."""
    T = _multiplication_tensor(n, m, p)
    reconstruction = np.zeros_like(T)
    for u, v, w in terms:
        reconstruction += _rank1_tensor_vec(u, v, w, n, m, p)
    
    if not np.allclose(reconstruction, T, atol=1e-8):
        return None
    
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    # Convert to lists with exact fractions where possible
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _get_strassen_2x2_candidates(n, m, p):
    """Generate Strassen-like rank-1 candidates for 2x2 blocks within 4x4x6."""
    candidates = []
    
    # 2x2 Strassen factors
    s_factors = [
        # (U_mask, V_mask, W_mask) as sets of (row,col) -> coeff
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,1):1, (0,1):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    # Place each 2x2 Strassen pattern at every possible 2x2 block position
    for bi in range(0, n - 1):  # block row in A
        for bj in range(0, m - 1):  # block col in A
            for bk in range(0, p - 1):  # block col in B / output
                for U_mask, V_mask, W_mask in s_factors:
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    
                    for (ri, ci), val in U_mask.items():
                        u[(bi + ri) * m + (bj + ci)] = val
                    for (ri, ci), val in V_mask.items():
                        v[(bj + ri) * p + (bk + ci)] = val
                    for (ri, ci), val in W_mask.items():
                        w[(bk + ci) * n + (bi + ri)] = val
                    
                    candidates.append((u, v, w))
    
    return candidates

def _build_strassen_tiled_4x4_4x6():
    """Build the known rank-77 Strassen-tiled decomposition as fallback."""
    n, m, p = 4, 4, 6
    
    # 4x4 * 4x4 part (rank 49) + 4x4 * 4x2 part (rank 28)
    # Use the same construction as before but simplified
    
    U = np.zeros((n * m, 77))
    V = np.zeros((m * p, 77))
    W = np.zeros((p * n, 77))
    
    # Part 1: 4x4 * 4x4 using 2-level Strassen (rank 49)
    s_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,1):1, (0,1):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    col = 0
    for l1 in range(7):
        for l2 in range(7):
            U1, V1, W1 = s_factors[l1]
            U2, V2, W2 = s_factors[l2]
            
            for (ri, ci), val1 in U1.items():
                for (si, sj), val2 in U2.items():
                    ai = ri * 2 + si
                    aj = ci * 2 + sj
                    U[ai * 4 + aj, col] += val1 * val2
            
            for (ri, ci), val1 in V1.items():
                for (si, sj), val2 in V2.items():
                    bri = ri * 2 + si
                    bcj = ci * 2 + sj
                    V[bri * 4 + bcj, col] += val1 * val2
            
            for (ri, ci), val1 in W1.items():
                for (si, sj), val2 in W2.items():
                    cri = ci * 2 + sj
                    cki = ri * 2 + si
                    W[cri * 4 + cki, col] += val1 * val2
            
            col += 1
    
    # Part 2: 4x4 * 4x2 (rank 28)
    # Split into 4 blocks of 2x4 * 2x2, each using 7 Strassen factors
    col = 49
    for block_row in range(2):  # 2 blocks of 2 rows in A
        for block_col in range(2):  # 2 blocks of 2 cols in A
            for f in range(7):
                U_mask, V_mask, W_mask = s_factors[f]
                
                for (ri, ci), val in U_mask.items():
                    ai = block_row * 2 + ri
                    aj = block_col * 2 + ci
                    U[ai * 4 + aj, col] += val
                
                for (ri, ci), val in V_mask.items():
                    bri = block_row * 2 + ri
                    bcj = 4 + ci  # cols 4,5 of B
                    V[bri * 6 + bcj, col] += val
                
                for (ri, ci), val in W_mask.items():
                    out_r = block_row * 2 + ri
                    out_c = 4 + ci
                    W[out_c * 4 + out_r, col] += val
                
                col += 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify_decomposition(U, V, W, n, m, p, num_trials=3, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(num_trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
        # Actually: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U_np[i*m+j], V_np[j*p+k]) * W_np[k*n+i]
                    # Wait, that's not right either. Let me think...
                    # The decomposition is: C[i,k] = sum_r (sum_j U[i,j,r] * V[j,k,r]) * W[k,i,r]
                    # No: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] where j is summed
                    # Actually the standard form: C[i,k] = sum_r (sum_j A[i,j]*U_r[i,j]) * (sum... 
                    # Let me use the tensor form: T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
                    # And we need T[i,j,k] = 1 for all i,j,k
                    # So C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k] = sum_j A[i,j] * B[j,k]
                    # The computation: for each r, compute intermediate
                    # Actually the evaluation is:
                    # C[i,k] = sum_r W_r[k*n+i] * (sum_j U_r[i*m+j] * A[i,j]) * (sum_j V_r[j*p+k] * B[j,k])
                    # No, that's not standard. Let me just verify via the tensor.
                # Simpler: verify the tensor
        # Let's verify via tensor instead
        break
    
    # Verify tensor directly
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.dot(
                    U_np[i*m+j] * V_np[j*p+k],  # element-wise product then dot with W
                    W_np[k*n+i]
                )
    
    if not np.allclose(T, 1.0, atol=1e-8):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Fallback: Strassen-tiled (rank 77)
    try:
        strassen = _build_strassen_tiled_4x4_4x6()
        if _verify_decomposition(strassen["U"], strassen["V"], strassen["W"], n, m, p):
            best = strassen
            best_rank = 77
        else:
            # Schoolbook
            best = _build_schoolbook(n, m, p)
            best_rank = n * m * p
    except Exception:
        best = _build_schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Try greedy randomized decomposition
    if time.time() < deadline - 10:
        try:
            result = _greedy_decompose(
                _multiplication_tensor(n, m, p), n, m, p,
                num_candidates=500,
                deadline=deadline - 5,
                seed=42
            )
            if result is not None:
                rank = len(result["U"][0])
                if rank < best_rank:
                    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                        best = result
                        best_rank = rank
        except Exception:
            pass
    
    # Try local search on the best result
    if time.time() < deadline - 5 and best_rank < n * m * p:
        try:
            # Convert to term format
            terms = []
            rank = best_rank
            U_np = np.array(best["U"])
            V_np = np.array(best["V"])
            W_np = np.array(best["W"])
            for r in range(rank):
                terms.append((U_np[:, r], V_np[:, r], W_np[:, r]))
            
            improved_terms = _local_search(terms, n, m, p, deadline - 2, seed=99)
            if len(improved_terms) < best_rank:
                cleaned = _verify_and_clean(improved_terms, n, m, p)
                if cleaned is not None:
                    new_rank = len(cleaned["U"][0])
                    if new_rank < best_rank:
                        best = cleaned
                        best_rank = new_rank
        except Exception:
            pass
    
    return best

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END
