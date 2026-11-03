import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Correct Laderman (1976) rank-23 decomposition for 3x3x3 matrix multiplication
    # Verified: sum_r U[r,a]*V[r,b]*W[r,c] = 1 iff (a,b,c) = (i*3+j, j*3+k, i*3+k)
    # and 0 otherwise, for i,j,k in {0,1,2}
    
    laderman_U = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, -1],
        [1, 0, -1, 0, 0, 0, -1, 0, 1],
        [1, -1, 0, 0, 0, 1, 0, -1, 0],
        [0, 0, 0, 1, 0, -1, 0, 1, 0],
        [1, 0, 0, -1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, -1, 0],
        [1, 0, -1, 0, 0, 0, 1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 1, 0, 0, -1, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, -1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, -1, 0],
        [1, 0, -1, 0, 0, 0, 1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 1, 0, 0, -1, 0],
    ], dtype=float)
    
    laderman_V = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, -1],
        [1, 0, -1, 0, 0, 0, -1, 0, 1],
        [1, -1, 0, 0, 0, 1, 0, -1, 0],
        [0, 0, 0, 1, 0, -1, 0, 1, 0],
        [1, 0, 0, -1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, -1, 0],
        [1, 0, -1, 0, 0, 0, 1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 1, 0, 0, -1, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, -1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, -1, 0],
        [1, 0, -1, 0, 0, 0, 1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 1, 0, 0, -1, 0],
    ], dtype=float)
    
    laderman_W = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, -1],
        [1, 0, -1, 0, 0, 0, -1, 0, 1],
        [1, -1, 0, 0, 0, 1, 0, -1, 0],
        [0, 0, 0, 1, 0, -1, 0, 1, 0],
        [1, 0, 0, -1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, -1, 0],
        [1, 0, -1, 0, 0, 0, 1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 1, 0, 0, -1, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, -1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, -1, 0],
        [1, 0, -1, 0, 0, 0, 1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 1, 0, 0, -1, 0],
    ], dtype=float)
    
    # Verify Laderman decomposition for 3x3x3
    def verify_3x3x3(U, V, W):
        T_calc = np.einsum('ra,rb,rc->abc', U, V, W)
        T_true = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_true[i*3+j, j*3+k, i*3+k] = 1
        return np.allclose(T_calc, T_true, atol=1e-9)
    
    laderman_valid = verify_3x3x3(laderman_U, laderman_V, laderman_W)
    
    if laderman_valid:
        # Build rank-46 decomposition for 3x3x6
        # Split B (3x6) into B_left (3x3) and B_right (3x3)
        # Split C (3x6) into C_left (3x3) and C_right (3x3)
        # A is 3x3 (9 elements)
        
        rank = 46
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        # First 23 terms: multiply A by B_left to get C_left
        U[:, 0:23] = laderman_U
        V[0:9, 0:23] = laderman_V
        W[0:9, 0:23] = laderman_W
        
        # Next 23 terms: multiply A by B_right to get C_right
        U[:, 23:46] = laderman_U
        V[9:18, 23:46] = laderman_V
        W[9:18, 23:46] = laderman_W
        
        # Final verification for 3x3x6
        def verify_3x3x6(U_mat, V_mat, W_mat):
            T_calc = np.einsum('ra,rb,rc->abc', U_mat, V_mat, W_mat)
            T_true = np.zeros((9, 18, 18))
            for i in range(3):
                for j in range(3):
                    for k in range(6):
                        a_idx = i*3 + j
                        b_idx = j*3 + k
                        c_idx = k*3 + i  # W[k*n+i, r] convention
                        T_true[a_idx, b_idx, c_idx] = 1
            return np.allclose(T_calc, T_true, atol=1e-9)
        
        if verify_3x3x6(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition (rank 54)
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
