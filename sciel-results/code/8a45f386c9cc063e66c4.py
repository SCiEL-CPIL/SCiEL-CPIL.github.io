"""Matrix multiplication 2x4x6 rank minimization via structured search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reproduces T."""
    T_check = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_check, T, atol=atol)

def _strassen_2x2x2():
    """
    Strassen rank-7 algorithm for 2x2 * 2x2 matrix multiplication.
    Returns list of 7 terms, each: (a_idx, b_idx, c_idx, d_idx)
    """
    return [
        (0, 3, 0, 3),    # m0: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        (2, 3, 0, 0),    # m1: (A[1,0]+A[1,1])*B[0,0]
        (0, 0, 1, 3),    # m2: A[0,0]*(B[0,1]-B[1,1])
        (3, 0, 2, 0),    # m3: A[1,1]*(B[1,0]-B[0,0])
        (0, 1, 3, 3),    # m4: (A[0,0]+A[0,1])*B[1,1]
        (2, 0, 0, 1),    # m5: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        (1, 3, 2, 3),    # m6: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    ]

def _strassen_output_coeffs():
    """
    For each Strassen term index (0..6), return dict of output (i,k) -> coefficient.
    """
    return [
        {(0,0): 1, (1,1): 1},    # m0
        {(1,0): 1, (1,1): -1},   # m1
        {(0,1): 1, (1,1): 1},    # m2
        {(0,0): 1, (1,0): 1},    # m3
        {(0,0): -1, (0,1): 1},   # m4
        {(1,1): 1},              # m5
        {(0,0): 1},              # m6
    ]

def _construct_strassen_42(n, m, p):
    """Construct the Strassen-based rank-42 decomposition."""
    strassen_terms = _strassen_2x2x2()
    strassen_out = _strassen_output_coeffs()

    all_terms = []  # (a1, a2, b1, b2, w_idx, coeff)

    for a_block in range(2):
        for b_block in range(2):
            for sub in range(3):
                a_base = a_block * 2
                b_base = b_block * 12 + sub * 2
                col_base = sub * 2

                for term_idx, (a_i, b_i, c_i, d_i) in enumerate(strassen_terms):
                    a1 = a_base + a_i
                    a2 = a_base + b_i
                    b1 = b_base + c_i
                    b2 = b_base + d_i

                    for (out_i, out_k), coeff in strassen_out[term_idx].items():
                        actual_k = col_base + out_k
                        w_idx = actual_k * n + out_i
                        all_terms.append((a1, a2, b1, b2, w_idx, coeff))

    rank = len(all_terms)
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)

    for col, (a1, a2, b1, b2, w_idx, coeff) in enumerate(all_terms):
        U[a1, col] = 1.0
        U[a2, col] = 1.0
        V[b1, col] = 1.0
        V[b2, col] = 1.0
        W[w_idx, col] = float(coeff)

    return U, V, W, rank

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _try_rank_reduction(U, V, W, T, max_rounds=50, deadline=None):
    """
    Try to reduce the rank by identifying linearly dependent columns
    and eliminating them. Returns (U, V, W, rank) with potentially lower rank.
    """
    U_dim, current_rank = U.shape
    V_dim = V.shape[0]
    W_dim = W.shape[0]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    for round_num in range(max_rounds):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        # Try to find a column that can be eliminated
        # A column r can be eliminated if its contribution can be absorbed
        # into other columns via linear combination
        
        # Check if any column is a linear combination of others
        # by looking at the tensor contribution
        
        eliminated = False
        for r in range(current_rank):
            if deadline is not None and time.time() > deadline - 2:
                break
            
            # Check if column r is redundant
            # Build the tensor contribution of column r
            # T_r = U[:,r] * V[:,r] * W[:,r]
            
            # Check if T_r is in the span of other columns
            # This is expensive, so use a heuristic:
            # If W[:,r] has only one non-zero entry, and that entry's
            # output index is covered by another column with compatible U,V,
            # we might be able to merge.
            
            # Simpler heuristic: check if removing column r still gives
            # a valid decomposition (i.e., the remaining columns span T)
            # This is expensive, so only try for small ranks.
            
            if current_rank <= 30:
                # Try removing column r
                U_test = np.delete(U, r, axis=1)
                V_test = np.delete(V, r, axis=1)
                W_test = np.delete(W, r, axis=1)
                
                if _verify(U_test, V_test, W_test, T, atol=1e-8):
                    U = U_test
                    V = V_test
                    W = W_test
                    current_rank -= 1
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = current_rank
                    eliminated = True
                    break
        
        if not eliminated:
            break
    
    return best_U, best_V, best_W, best_rank

def _try_merge_columns(U, V, W, T, deadline=None):
    """
    Try to merge pairs of columns that have overlapping support.
    Two columns can be merged if their W vectors are proportional
    and their U,V vectors can be combined.
    """
    U_dim, current_rank = U.shape
    V_dim = V.shape[0]
    W_dim = W.shape[0]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    for _ in range(20):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        merged = False
        for r1 in range(current_rank):
            if deadline is not None and time.time() > deadline - 2:
                break
            for r2 in range(r1 + 1, current_rank):
                if deadline is not None and time.time() > deadline - 2:
                    break
                
                # Check if W[:,r1] and W[:,r2] are proportional
                w1 = W[:, r1]
                w2 = W[:, r2]
                
                # Find non-zero entries
                nz1 = np.nonzero(w1)[0]
                nz2 = np.nonzero(w2)[0]
                
                if len(nz1) == 0 or len(nz2) == 0:
                    continue
                
                # Check if they have the same support
                if set(nz1.tolist()) != set(nz2.tolist()):
                    continue
                
                # Check if proportional
                ratio = None
                valid = True
                for idx in nz1:
                    if w1[idx] != 0:
                        if ratio is None:
                            ratio = w2[idx] / w1[idx]
                        elif abs(w2[idx] / w1[idx] - ratio) > 1e-9:
                            valid = False
                            break
                
                if not valid or ratio is None:
                    continue
                
                # Try to merge: new column has W = w1, U = U[:,r1] + ratio*U[:,r2], V = V[:,r1] + ratio*V[:,r2]
                # But this only works if the merged column's contribution equals the sum
                # Actually, we need: U1*V1*W1 + U2*V2*W2 = (U1 + alpha*U2)*(V1 + beta*V2)*W1
                # This requires specific conditions. Let's just try and verify.
                
                new_U = U[:, r1].copy()
                new_V = V[:, r1].copy()
                
                # The contribution of column r1 is U1*V1*W1
                # The contribution of column r2 is U2*V2*W2 = ratio * U2*V2*W1
                # So total = W1 * (U1*V1 + ratio*U2*V2)
                # We need U_new * V_new = U1*V1 + ratio*U2*V2
                # This is a rank-2 matrix (outer products sum), so U_new*V_new must have rank <= 2
                # But U_new and V_new are vectors, so U_new*V_new has rank 1.
                # This only works if U1*V1 + ratio*U2*V2 has rank 1.
                
                M = np.outer(U[:, r1], V[:, r1]) + ratio * np.outer(U[:, r2], V[:, r2])
                # Check if M has rank 1
                if M.size > 0:
                    # Quick rank check via SVD
                    try:
                        s = np.linalg.svd(M, compute_uv=False)
                        if len(s) > 1 and s[1] > 1e-8 * (s[0] if len(s) > 0 else 1):
                            continue
                    except:
                        continue
                
                # Extract U_new, V_new from M (rank 1)
                # M = u * v^T
                if M.size == 0:
                    continue
                # Find the largest magnitude entry
                max_idx = np.unravel_index(np.argmax(np.abs(M)), M.shape)
                max_val = M[max_idx]
                if abs(max_val) < 1e-12:
                    continue
                
                # Scale so that M[max_idx] = 1
                u_scaled = M[:, max_idx[1]] / max_val
                v_scaled = M[max_idx[0], :] / max_val
                
                # Verify
                if not np.allclose(np.outer(u_scaled, v_scaled), M, atol=1e-8):
                    continue
                
                # Build new matrices
                new_U_full = U[:, :].copy()
                new_V_full = V[:, :].copy()
                new_W_full = W[:, :].copy()
                
                # Replace column r1 with the merged column, remove column r2
                new_U_full[:, r1] = u_scaled
                new_V_full[:, r1] = v_scaled
                # W stays the same for r1
                
                # Remove column r2
                if r2 < current_rank - 1:
                    new_U_full = np.delete(new_U_full, r2, axis=1)
                    new_V_full = np.delete(new_V_full, r2, axis=1)
                    new_W_full = np.delete(new_W_full, r2, axis=1)
                else:
                    new_U_full = new_U_full[:, :r2]
                    new_V_full = new_V_full[:, :r2]
                    new_W_full = new_W_full[:, :r2]
                
                if _verify(new_U_full, new_V_full, new_W_full, T, atol=1e-8):
                    U, V, W = new_U_full, new_V_full, new_W_full
                    current_rank -= 1
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = current_rank
                    merged = True
                    break
            
            if merged:
                break
        
        if not merged:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks as starting point, then attempts rank reduction.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor for verification
    T = _build_tensor(n, m, p)

    # Start with Strassen rank-42 decomposition
    U, V, W, rank = _construct_strassen_42(n, m, p)
    
    # Verify the Strassen decomposition
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # Try to reduce rank via column merging
    if time.time() < deadline - 10:
        U2, V2, W2, rank2 = _try_merge_columns(U, V, W, T, deadline=deadline)
        if rank2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, rank2

    # Try to reduce rank via linear dependency elimination
    if time.time() < deadline - 10:
        U3, V3, W3, rank3 = _try_rank_reduction(best_U, best_V, best_W, T, max_rounds=30, deadline=deadline)
        if rank3 < best_rank:
            best_U, best_V, best_W, best_rank = U3, V3, W3, rank3

    # Final verification
    if not _verify(best_U, best_V, best_W, T):
        # Fallback to schoolbook
        best_U, best_V, best_W, best_rank = _construct_schoolbook(n, m, p)

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
