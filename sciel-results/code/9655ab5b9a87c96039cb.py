import numpy as np
import time

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

def _validate(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    # Optimization: Use numpy for validation
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check all 1 positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # s = sum_r U[a,r] V[b,r] W[c,r]
                s = np.dot(np.dot(U_np[a, :], V_np[b, :]), W_np[c, :])
                if abs(s - 1.0) > 1e-9:
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
        
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    try:
        U, V, W = _construct_strassen_4x8()
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
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
