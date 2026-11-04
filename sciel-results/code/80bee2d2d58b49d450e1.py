import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 130  # seconds, with safety margin

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (local 2x2)
    strassen_A = [
        [1, 0, 0, 1], # M0
        [0, 0, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 0, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0], # M5
        [0, 1, 0, -1]  # M6
    ]
    
    # Strassen coefficients for B (local 2x2)
    strassen_B = [
        [1, 0, 0, 1], # M0
        [1, 0, 0, 0], # M1
        [0, 1, 0, -1], # M2
        [-1, 0, 1, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, 1]  # M6
    ]
    
    # W coefficients: w[t][i*2 + k_loc]
    w_coeffs = [
        [1, 0, 0, 1], # t=0
        [0, 0, 1, -1], # t=1
        [0, 1, 0, 1], # t=2
        [1, 0, 1, 0], # t=3
        [-1, 1, 0, 0], # t=4
        [0, 0, 0, 1], # t=5
        [1, 0, 0, 0]  # t=6
    ]
    
    # 6 blocks: b = 3*s + k_block
    # s in 0..1, k_block in 0..2
    for b in range(6):
        s = b // 3
        k_block = b % 3
        
        for t in range(7):
            c = 7 * b + t
            
            # Fill U
            u_coeff = strassen_A[t]
            for i in range(2):
                for j_loc in range(2):
                    idx_A = i * 2 + j_loc
                    coeff = u_coeff[idx_A]
                    if coeff != 0:
                        global_i = i
                        global_j = 2 * s + j_loc
                        U[global_i * m + global_j][c] = coeff
                        
            # Fill V
            v_coeff = strassen_B[t]
            for j_loc in range(2):
                for k_loc in range(2):
                    idx_B = j_loc * 2 + k_loc
                    coeff = v_coeff[idx_B]
                    if coeff != 0:
                        global_j = 2 * s + j_loc
                        global_k = 2 * k_block + k_loc
                        V[global_j * p + global_k][c] = coeff
                        
            # Fill W
            w_coeff = w_coeffs[t]
            for i in range(2):
                for k_loc in range(2):
                    idx_W = i * 2 + k_loc
                    coeff = w_coeff[idx_W]
                    if coeff != 0:
                        global_k = 2 * k_block + k_loc
                        global_i = i
                        W[global_k * n + global_i][c] = coeff
                        
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify the decomposition is correct
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Check against standard matmul tensor
    def verify_decomp(U_arr, V_arr, W_arr):
        """Verify that the decomposition gives the correct matmul tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Expected: T[i,j,k] = 1 if this is a valid matmul position, else 0
                    # Actually, the matmul tensor T_{i,j,k} = delta_{i,j,k} doesn't make sense for rectangular
                    # The tensor is: T[i,j,k] = 1 for all (i,j,k) since matmul(A,B)[i,k] = sum_j A[i,j]*B[j,k]
                    # So the tensor entries are all 1? No, that's not right either.
                    
                    # The multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1 for all i,j,k
                    # because (AB)_{ik} = sum_j A_{ij} B_{jk}, so each term A_{ij}B_{jk} contributes to (AB)_{ik}
                    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * 1
                    # So T[i,j,k] = 1 for all i,j,k
                    
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_np, V_np, W_np):
        # Fallback: return schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Try to reduce rank by finding linear dependencies
    # The idea: if we can express some rank-1 terms as linear combinations of others,
    # we might be able to reduce the total rank.
    
    # Approach: Try to find a subset of columns that still spans the tensor
    # This is NP-hard in general, so we'll do a greedy/heuristic approach
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 30:
        # Not enough time for search, return the rank-42 solution
        return {"U": U, "V": V, "W": W}
    
    # Try rank reduction: check if any column is a linear combination of others
    # For each column c, check if U[:,c] is in the span of U[:,c'] for c' != c
    # AND V[:,c] is in the span of V[:,c'] for c' != c  
    # AND W[:,c] is in the span of W[:,c'] for c' != c
    # If all three hold, we might be able to eliminate that column
    
    # More sophisticated: try to find a transformation that reduces the rank
    # We'll try random projections and check if the tensor is preserved
    
    best_rank = rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try eliminating columns one at a time (greedy)
    columns_to_keep = list(range(rank))
    
    for attempt in range(100):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        # Try to remove a random column
        c_to_remove = columns_to_keep[attempt % len(columns_to_keep)]
        
        # Check if we can express column c_to_remove as a combination of others
        # This is complex, so we'll just try removing it and see if the tensor is still decomposed
        # by re-solving for the remaining columns
        
        # Simpler approach: just try smaller random subsets
        if attempt % 10 == 0:
            # Try a random subset of size rank - 2
            subset_size = rank - 2
            subset = list(np.random.choice(rank, subset_size, replace=False))
            
            # Check if the subset still decomposes the tensor
            # We need to check if there exist coefficients such that the tensor is preserved
            # This is a system of equations: for each (i,j,k), sum_{c in subset} U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # But we can't change U,V,W, only select columns
            # So we just check if the selected columns already give the right answer
            # (which they won't unless the removed columns had zero contribution)
            pass
    
    # For now, return the rank-42 solution
    return {"U": U, "V": V, "W": W}
