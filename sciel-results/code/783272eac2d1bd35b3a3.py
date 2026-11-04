import time
import numpy as np
from itertools import combinations, product

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Return True iff the decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    # Target is T[i,j,k] = 1
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    # Check no extra terms outside the valid range? 
    # The tensor shape is (n, m, p). Einsum produces exactly that.
    # However, we must ensure no "ghost" terms if dimensions were larger, 
    # but here shapes match exactly.
    return True

def _get_supports(n, m, p, max_support=16):
    """
    Generate a dictionary of possible rank-1 support patterns.
    A rank-1 term is defined by three vectors u, v, w.
    The support is the set of indices (i, j, k) where u[i]v[j]w[k] != 0.
    We categorize patterns by the set of non-zero indices in U, V, W.
    
    To keep the search space manageable, we assume the rank-1 terms are "sparse".
    Specifically, we look for patterns where the number of non-zeros in U, V, W is small.
    
    Actually, a better approach for ILP is to pre-compute a set of "candidate" rank-1 tensors 
    that have integer/half-integer coefficients and small support.
    
    Let's define a candidate by its non-zero structure.
    For a 4x4x9 tensor, the maximum support is 144.
    We will generate candidates with support size <= 16.
    """
    candidates = []
    # Indices for U: 0..15 (4x4)
    # Indices for V: 0..35 (4x9)
    # Indices for W: 0..35 (9x4)
    
    # We will iterate over small subsets of indices for U, V, W.
    # If we pick a subset I for U, J for V, K for W, the support is the "box" product?
    # No, a rank-1 term u \otimes v \otimes w has support = supp(u) x supp(v) x supp(w) 
    # ONLY if we consider the tensor product structure. 
    # Wait, the tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # A single rank-1 term r contributes U[i,j] V[j,k] W[k,i] to T[i,j,k].
    # The support of this contribution is NOT a simple box product in (i,j,k) space 
    # because the indices are coupled: U depends on (i,j), V on (j,k), W on (k,i).
    # 
    # Let S_U be the set of (i,j) where U[i,j] != 0.
    # Let S_V be the set of (j,k) where V[j,k] != 0.
    # Let S_W be the set of (k,i) where W[k,i] != 0.
    # 
    # The contribution to T[i,j,k] is non-zero iff:
    # 1. (i,j) in S_U
    # 2. (j,k) in S_V
    # 3. (k,i) in S_W
    # 
    # This defines a specific pattern of non-zeros in the 4x4x9 tensor.
    # We can enumerate small sets S_U, S_V, S_W.
    
    # To reduce complexity, we restrict S_U, S_V, S_W to have size <= 4.
    # Number of subsets of size 4 from 16 is C(16,4) = 1820.
    # Number of subsets of size 4 from 36 is C(36,4) = 58905.
    # This is too large to enumerate all triples.
    
    # Alternative Strategy: 
    # Instead of enumerating all patterns, we use a "Greedy Set Cover" with randomization 
    # or a small ILP over a restricted set of "useful" patterns.
    # 
    # Let's try a different structural decomposition first: 
    # The 4x4x9 multiplication can be viewed as 9 columns of 4x4 matrices.
    # Each column k corresponds to a 4x4 matrix multiplication A * e_k (where e_k is a column of B).
    # Actually, it's A (4x4) * B (4x9).
    # We can decompose this into 9 separate 4x4x1 matrix-vector products? No, that's rank 16*9=144.
    # 
    # Let's look at the "Middle Product" hint.
    # Rank 4x4x4 is 36 (Laderman).
    # Can we do better for 4x4x9?
    # 
    # Let's implement a simple ILP solver using scipy.optimize.milp if available, 
    # or a custom branch-and-bound if not. 
    # Given the 150s limit, a generic ILP might be too slow if the variable space is huge.
    
    # Revised Strategy: 
    # Use the known Laderman 36-rank decomposition for 4x4x4.
    # For 4x4x9, we can split B into two parts: B1 (4x4) and B2 (4x5).
    # A*B1 is 4x4x4 (Rank 36).
    # A*B2 is 4x4x5.
    # What is the rank of 4x4x5?
    # It is at most 4*4*5 = 80 (schoolbook).
    # Using Strassen-like methods, 4x4x2 is rank 12? (2*2*2 is 7, so 4*4*2 is 7*2=14? No, 4x4x2 is 2x 2x4x2? No.)
    # 4x4x1 is rank 16.
    # 4x4x2 is rank 12? Let's check. 2x2x2 is 7. 4x4x2 = 2 * (2x4x2)? No.
    # 4x4x2 can be done with 12 multiplications? 
    # Actually, 4x4x2 is equivalent to two 4x4x1. Rank 16*2 = 32? 
    # No, there are better ways. 
    # 
    # Let's try to construct a rank-52 solution manually using blocks.
    # Block 1: 4x4x4 (Laderman, rank 36).
    # Block 2: 4x4x5.
    # If we can find a rank-16 decomposition for 4x4x5, total rank = 36+16=52.
    # Is rank(4x4x5) <= 16?
    # 4x4x1 is rank 16. 
    # 4x4x5 is 5 such vectors. If we do them independently, rank is 5*16 = 80.
    # But we can share computations.
    # 
    # Actually, the rank of 4x4xn is known to be roughly 4n for small n?
    # For n=1, rank=16.
    # For n=4, rank=36. (16 + 20)
    # For n=9, rank=?
    # 
    # Let's assume the rank of 4x4x5 is 20? Then total 56.
    # 
    # Since I cannot derive the exact Laderman 4x4x5 decomposition from scratch in this thought process 
    # with guaranteed correctness for the ILP, I will implement a **Randomized Greedy Search** 
    # over the space of integer/half-integer rank-1 terms.
    
    # We will generate random rank-1 terms with small support (e.g., support size 4-8 in U, V, W).
    # We will try to cover the target tensor T=1 using a linear combination of these terms.
    # This is a linear system A x = b. 
    # We want to find a sparse x (min number of non-zeros).
    # We can use L1-regularized least squares or a greedy matching pursuit.
    
    # Step 1: Generate a library of candidate rank-1 terms.
    # Step 2: Use Matching Pursuit to select the best terms.
    
    pass

def _generate_candidate_terms(n, m, p, num_candidates=5000):
    """
    Generate random rank-1 terms with small support.
    """
    candidates = []
    # We want terms where U, V, W have few non-zeros.
    # Let's try support sizes 2, 3, 4 for U, V, W.
    for _ in range(num_candidates):
        # Random support for U (4x4)
        su_size = np.random.randint(1, 5)
        su = np.random.choice(16, su_size, replace=False)
        # Random support for V (4x9)
        sv_size = np.random.randint(1, 5)
        sv = np.random.choice(36, sv_size, replace=False)
        # Random support for W (9x4)
        sw_size = np.random.randint(1, 5)
        sw = np.random.choice(36, sw_size, replace=False)
        
        # Check if this combination actually contributes to the tensor
        # We need to ensure that the "overlap" of indices allows T[i,j,k] to be formed.
        # A term contributes to T[i,j,k] if:
        # (i,j) in S_U, (j,k) in S_V, (k,i) in S_W.
        # We can just create the vectors and check if the resulting tensor is non-zero.
        
        u = np.zeros(16)
        v = np.zeros(36)
        w = np.zeros(36)
        
        # Assign random integer coefficients (1 or -1)
        for idx in su: u[idx] = np.random.choice([-1, 1])
        for idx in sv: v[idx] = np.random.choice([-1, 1])
        for idx in sw: w[idx] = np.random.choice([-1, 1])
        
        # Reshape to check support in tensor
        u_mat = u.reshape(4, 4)
        v_mat = v.reshape(4, 9)
        w_mat = w.reshape(9, 4)
        
        # Compute the tensor contribution
        # T[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        t_contrib = np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        
        if np.any(t_contrib != 0):
            candidates.append((u, v, w, t_contrib))
            
    return candidates

def _solve_sparse_system(candidates, n, m, p, max_iter=1000):
    """
    Use Matching Pursuit to find a sparse combination of candidates that sums to T=1.
    """
    target = np.ones((n, m, p))
    residual = target.copy()
    selected_indices = []
    coefficients = []
    
    # Pre-flatten candidates for speed
    cand_tensors = np.array([c[3] for c in candidates]).reshape(len(candidates), -1)
    target_flat = target.flatten()
    residual_flat = residual.flatten()
    
    # Normalize candidates? Not strictly necessary for exact integer solutions, 
    # but helps in matching pursuit.
    # Since we want integer/half-integer solutions, we should be careful.
    # Matching pursuit finds real coefficients. We can then try to round them.
    
    for _ in range(max_iter):
        if np.max(np.abs(residual_flat)) < 1e-9:
            break
            
        # Find candidate most correlated with residual
        correlations = cand_tensors @ residual_flat
        # Pick max absolute correlation
        idx = np.argmax(np.abs(correlations))
        
        # Avoid picking the same candidate twice unless necessary (usually not for exact cover)
        if idx in selected_indices:
             # If we pick it again, we are just adjusting the coefficient.
             # For simplicity in this heuristic, we allow it but it might loop.
             # Better: restrict to unselected? 
             # If we restrict, we might miss solutions needing multiple of the same pattern.
             # However, for matrix multiplication, distinct patterns are usually needed.
             # Let's try to pick the best *unselected* one first.
             mask = np.ones(len(candidates), dtype=bool)
             for s_idx in selected_indices:
                 mask[s_idx] = False
             if np.any(mask):
                 corr_masked = correlations.copy()
                 corr_masked[~mask] = -np.inf
                 idx = np.argmax(corr_masked)
             else:
                 break # No more unique candidates
        
        # Calculate coefficient
        # coeff = <residual, cand> / <cand, cand>
        cand_vec = cand_tensors[idx]
        dot_rr = np.dot(cand_vec, cand_vec)
        if dot_rr == 0:
            continue
        coeff = np.dot(residual_flat, cand_vec) / dot_rr
        
        # Update residual
        residual_flat -= coeff * cand_vec
        residual = residual_flat.reshape(n, m, p)
        
        selected_indices.append(idx)
        coefficients.append(coeff)
        
        # Check convergence
        if np.max(np.abs(residual_flat)) < 1e-6:
            break
            
    if np.max(np.abs(residual_flat)) > 1e-5:
        return None, None, None, None

    # Now we have a set of indices and real coefficients.
    # We need to construct U, V, W matrices.
    # Since the coefficients might be fractional, we might need to scale.
    # However, the task allows half-integers.
    # Let's check if the coefficients are close to integers or half-integers.
    
    U = np.zeros((n*m, len(selected_indices)))
    V = np.zeros((m*p, len(selected_indices)))
    W = np.zeros((p*n, len(selected_indices)))
    
    for k, (idx, coeff) in enumerate(zip(selected_indices, coefficients)):
        u, v, w, _ = candidates[idx]
        U[:, k] = u * coeff
        V[:, k] = v * coeff
        W[:, k] = w * coeff
        
    return U, V, W, len(selected_indices)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Generate a large pool of candidate rank-1 terms
    # We need a diverse pool.
    # Let's increase the pool size and diversity.
    candidates = _generate_candidate_terms(n, m, p, num_candidates=20000)
    
    if not candidates:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Solve sparse system
    U, V, W, rank_found = _solve_sparse_system(candidates, n, m, p, max_iter=5000)
    
    if U is not None and _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 3. Fallback: Try a known structured decomposition if the random search fails.
    # We will use the 114-rank decomposition from the seed as a fallback,
    # but first, let's try a slightly better structured one if time permits.
    # For now, return the seed's logic to ensure validity.
    
    # Re-implementing the seed's rank-114 logic as fallback
    _STRASSEN_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    def _build_rank114(n, m, p):
        rank = 114
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for (u, v, w) in _STRASSEN_222:
            for t in range(4):
                if u[t]:
                    U[t][col] += u[t]
                if v[t]:
                    V[t][col] += v[t]
                if w[t]:
                    W[t][col] += w[t]
            col += 1
        for (u, v, w) in _STRASSEN_222:
            for t in range(4):
                if u[t]:
                    U[t][col] += u[t]
                if v[t]:
                    V[4 + t][col] += v[t]
                if w[t]:
                    W[4 + t][col] += w[t]
            col += 1
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[8 + j][col] = 1
                W[8 + i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}

    res = _build_rank114(n, m, p)
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    if not _verify(U, V, W, n, m, p):
        # Ultimate fallback: Schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Quick test
    res = construct()
    print("Rank:", len(res['U'][0]))
