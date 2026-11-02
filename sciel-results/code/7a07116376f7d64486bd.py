import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 8]

def _get_strassen_factors(n):
    """
    Returns U, V, W for n x n x n matrix multiplication using Strassen.
    """
    if n == 1:
        return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
    
    n_half = n // 2
    U_half, V_half, W_half = _get_strassen_factors(n_half)
    rank_half = U_half.shape[1]
    
    U = np.zeros((n*n, 7 * rank_half))
    V = np.zeros((n*n, 7 * rank_half))
    W = np.zeros((n*n, 7 * rank_half))
    
    A_coeffs = [
        {0: 1, 3: 1}, 
        {2: 1, 3: 1}, 
        {0: 1},       
        {3: 1},       
        {0: 1, 1: 1}, 
        {2: 1, 0: -1},
        {1: 1, 3: -1} 
    ]
    
    B_coeffs = [
        {0: 1, 3: 1}, 
        {0: 1},       
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},       
        {0: 1, 1: 1}, 
        {2: 1, 3: 1}  
    ]
    
    C_coeffs = [
        {0: 1, 3: 1}, 
        {2: 1, 3: -1},
        {1: 1, 3: 1}, 
        {0: 1, 2: 1}, 
        {0: -1, 1: 1},
        {3: 1},       
        {0: 1}        
    ]
    
    for f in range(7):
        col_offset = f * rank_half
        for l in range(rank_half):
            col = col_offset + l
            
            for b_idx, coeff in A_coeffs[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                for i in range(n_half):
                    for j in range(n_half):
                        local_idx = i * n_half + j
                        global_idx = (r_start + i) * n + (c_start + j)
                        U[global_idx, col] += coeff * U_half[local_idx, l]
            
            for b_idx, coeff in B_coeffs[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                for i in range(n_half):
                    for j in range(n_half):
                        local_idx = i * n_half + j
                        global_idx = (r_start + i) * n + (c_start + j)
                        V[global_idx, col] += coeff * V_half[local_idx, l]
            
            for b_idx, coeff in C_coeffs[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                for k_local in range(n_half):
                    for i_local in range(n_half):
                        local_idx = k_local * n_half + i_local
                        global_idx = (r_start + k_local) * n + (c_start + i_local)
                        W[global_idx, col] += coeff * W_half[local_idx, l]
                        
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(np.dot(U_np[a, :], V_np[b, :]), W_np[c, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_strassen_4x8():
    n, m, p = 4, 4, 8
    
    U4, V4, W4 = _get_strassen_factors(4)
    rank4 = U4.shape[1]
    
    U = np.zeros((n*m, 2*rank4))
    V = np.zeros((m*p, 2*rank4))
    W = np.zeros((p*n, 2*rank4))
    
    # Left half
    U[:, 0:rank4] = U4
    V[0:16, 0:rank4] = V4
    W[0:16, 0:rank4] = W4
    
    # Right half
    U[:, rank4:2*rank4] = U4
    
    # Map V4 to V right half
    for r in range(16):
        j_local = r // 4
        k_local = r % 4
        v_row_in_V = j_local * 4 + (k_local + 4)
        V[v_row_in_V, rank4:2*rank4] = V4[r, :]
        
    # Map W4 to W right half
    for r in range(16):
        k_local = r // 4
        i_local = r % 4
        w_row_in_W = (k_local + 4) * 4 + i_local
        W[w_row_in_W, rank4:2*rank4] = W4[r, :]
        
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank by finding linear dependencies.
    Uses the fact that we need U[a,r]*V[b,r]*W[c,r] = 1 for valid (a,b,c) triples
    and 0 otherwise.
    """
    start_rank = U.shape[1]
    if start_rank <= target_rank:
        return None
    
    # Build the constraint matrix: for each (i,j,k) triple, we need
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For invalid triples, it should be 0.
    
    # Approach: try to find a subset of columns that spans the same "effective" space
    # by checking if we can express some columns as linear combinations of others
    # while preserving the tensor decomposition.
    
    # Simpler approach: use random projection to try to find a lower-rank equivalent
    # This is a heuristic - we try random linear combinations of columns
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try to find dependencies using SVD-like approach
    # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
    # We want to find if some columns are redundant.
    
    # Method: For each pair of columns, check if one can be removed
    # by adjusting the others. This is expensive, so use a simpler heuristic.
    
    # Heuristic: check if any column is a linear combination of others
    # in the "effective" sense (preserving the tensor).
    
    # Alternative: use the fact that for rectangular matrices, there may be
    # additional structure. Try to find a basis for the span of {u_r} and see
    # if we can re-express the decomposition with fewer terms.
    
    # For now, try a simple approach: check if removing any single column
    # still gives a valid decomposition (this won't work directly since we need
    # exact equality, but let's try random perturbations).
    
    # Better approach: use linear algebra to find the minimal rank.
    # The matrix multiplication tensor for n×m × m×p has a known lower bound.
    # For 4×4 × 4×8, the schoolbook rank is 4*4*8 = 128.
    # Strassen gives 2 * 49 = 98 for this case.
    # Let's try to see if we can do better.
    
    # Try: use the fact that the two 4×4 blocks share the same U factors.
    # This means U has rank at most 49 (from one block), not 98.
    # So we might be able to reduce the effective rank.
    
    # Check: is U[:, 0:rank4] == U[:, rank4:2*rank4]?
    if np.allclose(U[:, 0:start_rank//2], U[:, start_rank//2:]):
        # U is duplicated, so we can potentially reduce rank
        # by finding a common basis for the two V and W blocks.
        
        # The constraint is:
        # sum_r U[a,r] * (V_left[b,r] * W_left[c,r] + V_right[b,r] * W_right[c,r]) = 1
        # for valid triples.
        
        # This is equivalent to:
        # sum_r U[a,r] * V_combined[b,r] * W_combined[c,r] = 1
        # where V_combined and W_combined are constructed appropriately.
        
        # Actually, the decomposition is:
        # T = sum_r U[a,r] V_left[b,r] W_left[c,r] + sum_r U[a,r] V_right[b,r] W_right[c,r]
        #   = sum_r U[a,r] (V_left[b,r] W_left[c,r] + V_right[b,r] W_right[c,r])
        
        # This is not a standard rank-r decomposition unless we can combine
        # the two terms into a single product for each r.
        
        # Let's try: define new factors such that
        # V_new[b, r'] and W_new[c, r'] satisfy
        # V_new[b, r'] * W_new[c, r'] = V_left[b, r'] * W_left[c, r'] + V_right[b, r'] * W_right[c, r']
        # for r' in some range.
        
        # This is generally not possible with the same number of terms,
        # but let's try to find a lower-rank approximation.
        
        pass
    
    # For now, return the original if we can't find a reduction
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    try:
        U, V, W = _construct_strassen_4x8()
        if _validate(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
        
    # Fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
