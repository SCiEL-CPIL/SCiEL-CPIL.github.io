"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_raw_decomposition(n, m, p):
    """
    Builds a raw (possibly redundant) decomposition for 4x4 x 4x6 matrix multiplication.
    Uses tensor product of Strassen 2x2x2 (rank 7) for the 4x4x4 part (rank 49)
    and Schoolbook for the 4x4x2 part (rank 32).
    Returns list of (U_dict, V_dict, W_dict) tuples.
    """
    strassen_terms_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    rank_terms = []
    
    for s in range(7):
        u_s, v_s, w_s = strassen_terms_2x2[s]
        for t in range(7):
            u_t, v_t, w_t = strassen_terms_2x2[t]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
                    
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
                    
            for (i1, k1), cw in w_s.items():
                for (i2, k2), cw2 in w_t.items():
                    gi = 2 * i1 + i2
                    gk = 2 * k1 + k2
                    coeff = cw * cw2
                    key = (gk, gi)
                    W_global[key] = W_global.get(key, 0) + coeff
                    
            rank_terms.append((U_global, V_global, W_global))
            
    for i in range(4):
        for j in range(4):
            for k in range(2):
                gk = 4 + k
                U_global = {(i, j): 1}
                V_global = {(j, gk): 1}
                W_global = {(gk, i): 1}
                rank_terms.append((U_global, V_global, W_global))
                
    return rank_terms

def _reduce_decomposition(rank_terms, n, m, p, tol=1e-10):
    """
    Reduces the rank of a decomposition by finding linear dependencies among the rank-1 terms.
    Uses the fact that if sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 as a tensor, then we can eliminate
    one term. We detect this by checking if the coefficient vectors are linearly dependent
    when projected onto the appropriate spaces.
    
    Strategy: Build the full coefficient matrices and use SVD to find a minimal spanning set.
    """
    num_terms = len(rank_terms)
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Build coefficient matrices
    U_mat = np.zeros((U_size, num_terms), dtype=float)
    V_mat = np.zeros((V_size, num_terms), dtype=float)
    W_mat = np.zeros((W_size, num_terms), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U_mat[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V_mat[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W_mat[k * n + i, r] = coeff
    
    # The decomposition is valid iff for all i,j,k:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = C[i,k] where C is the matrix product
    # 
    # To reduce rank, we need to find a linear transformation that preserves the tensor.
    # A simpler approach: use the fact that the tensor T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # is fixed. If we have redundant terms, we can find a lower-rank representation.
    #
    # Practical approach: Try to find a subset of terms that still spans the same tensor.
    # We do this by checking linear independence of the "slices" of the tensor.
    
    # Alternative: Use the Kronecker product structure. The tensor can be viewed as
    # a 3D array T[i,j,k]. We can compute its rank via unfolding.
    
    # Let's try a greedy approach: start with all terms, try to remove each one and check
    # if the remaining terms still give the correct tensor.
    
    # First, compute the target tensor
    # T[i,j,k] should equal sum over the decomposition
    # Actually, let's just verify correctness and try to find minimal subsets.
    
    # For now, use SVD-based reduction on the coefficient matrices.
    # The key insight: if the columns of [U; V; W] (stacked) are linearly dependent,
    # we can reduce the rank.
    
    # Stack the three matrices
    stacked = np.vstack([U_mat, V_mat, W_mat])
    n_rows, n_cols = stacked.shape
    
    # Compute SVD
    U_svd, s_vals, Vt_svd = np.linalg.svd(stacked, full_matrices=False)
    
    # Determine effective rank
    effective_rank = np.sum(s_vals > tol * s_vals[0])
    
    # Extract the top `effective_rank` right singular vectors
    if effective_rank < num_terms:
        # The new coefficient matrices are the top effective_rank rows of Vt_svd
        # But we need to be careful: this gives a projection, not an exact reduction.
        
        # For exact reduction, we need the null space.
        # Find the null space of the stacked matrix
        null_space = Vt_svd[effective_rank:]
        null_dim = null_space.shape[0]
        
        # We can express the last null_dim terms as linear combinations of the first
        # (num_terms - null_dim) terms.
        
        # New U, V, W are just the first (num_terms - null_dim) columns
        keep = num_terms - null_dim
        U_new = U_mat[:, :keep]
        V_new = V_mat[:, :keep]
        W_new = W_mat[:, :keep]
        
        # Verify this reduced decomposition is still correct
        if _verify(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
    
    # If SVD reduction doesn't work (or gives no improvement), return original
    return U_mat, V_mat, W_mat

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0
    
    # Build raw decomposition
    rank_terms = _build_raw_decomposition(n, m, p)
    
    # Convert to matrices
    num_terms = len(rank_terms)
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    U_mat = np.zeros((U_size, num_terms), dtype=float)
    V_mat = np.zeros((V_size, num_terms), dtype=float)
    W_mat = np.zeros((W_size, num_terms), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U_mat[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V_mat[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W_mat[k * n + i, r] = coeff
    
    # Try to reduce rank
    U_reduced, V_reduced, W_reduced = _reduce_decomposition(rank_terms, n, m, p)
    
    # Verify reduced version
    if _verify(U_reduced, V_reduced, W_reduced, n, m, p):
        U, V, W = U_reduced, V_reduced, W_reduced
    else:
        # Fallback to original
        U, V, W = U_mat, V_mat, W_mat
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
