import time
import numpy as np
import random

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140
    
    n, m, p = DIMENSIONS
    
    # Initialize with Strassen Rank 28
    R = 28
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Helper to generate Strassen block
    def strassen_block():
        U_loc = np.zeros((2, 2, 7))
        V_loc = np.zeros((2, 2, 7))
        W_loc = np.zeros((2, 2, 7))
        
        # r=0 (M1)
        U_loc[0, 0, 0] = 1; U_loc[1, 1, 0] = 1
        V_loc[0, 0, 0] = 1; V_loc[1, 1, 0] = 1
        W_loc[0, 0, 0] = 1; W_loc[1, 0, 0] = 1; W_loc[0, 1, 0] = 1; W_loc[1, 1, 0] = -1
        
        # r=1 (M2)
        U_loc[1, 0, 1] = 1; U_loc[1, 1, 1] = 1
        V_loc[0, 0, 1] = 1
        W_loc[0, 1, 1] = 1; W_loc[1, 1, 1] = -1
        
        # r=2 (M3)
        U_loc[0, 0, 2] = 1
        V_loc[1, 0, 2] = 1; V_loc[1, 1, 2] = -1
        W_loc[1, 0, 2] = 1; W_loc[1, 1, 2] = 1
        
        # r=3 (M4)
        U_loc[1, 1, 3] = 1
        V_loc[0, 1, 3] = 1; V_loc[0, 0, 3] = -1
        W_loc[0, 0, 3] = 1; W_loc[0, 1, 3] = 1
        
        # r=4 (M5)
        U_loc[0, 0, 4] = 1; U_loc[0, 1, 4] = 1
        V_loc[1, 1, 4] = 1
        W_loc[0, 0, 4] = -1; W_loc[1, 0, 4] = 1
        
        # r=5 (M6)
        U_loc[1, 0, 5] = 1; U_loc[0, 0, 5] = -1
        V_loc[0, 0, 5] = 1; V_loc[1, 0, 5] = 1
        W_loc[1, 1, 5] = 1
        
        # r=6 (M7)
        U_loc[0, 1, 6] = 1; U_loc[1, 1, 6] = -1
        V_loc[0, 1, 6] = 1; V_loc[1, 1, 6] = 1
        W_loc[0, 0, 6] = 1
        
        return U_loc, V_loc, W_loc

    U_loc, V_loc, W_loc = strassen_block()
    
    blocks = [
        (0, 0, 0),
        (2, 0, 0),
        (0, 2, 2),
        (2, 2, 2),
    ]
    
    for b_idx, (a_off, b_off, c_off) in enumerate(blocks):
        r_offset = b_idx * 7
        for i in range(2):
            for j in range(2):
                U[i * m + (a_off + j), r_offset : r_offset + 7] = U_loc[i, j, :]
        for j in range(2):
            for k in range(2):
                V[(b_off + j) * p + (c_off + k), r_offset : r_offset + 7] = V_loc[j, k, :]
        for k in range(2):
            for i in range(2):
                W[(c_off + k) * n + i, r_offset : r_offset + 7] = W_loc[k, i, :]

    def check_bilinear(U, V, W, trials=5):
        for _ in range(trials):
            A = np.random.randint(-10, 10, (n, m))
            B = np.random.randint(-10, 10, (m, p))
            C_expected = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(U.shape[1]):
                # u_r is a vector of length n*m
                # v_r is a vector of length m*p
                # w_r is a vector of length p*n (indexed k*n+i)
                
                # Sum A part: sum_{i',j'} U[i',j',r] A[i',j']
                # U is flattened i*m+j
                A_proj = np.sum(U[:, r] * A.flatten())
                B_proj = np.sum(V[:, r] * B.flatten())
                
                # Add to C
                # W is flattened k*n+i
                # C[i,k] += W[k*n+i, r] * A_proj * B_proj
                C_calc += W[:, r].reshape(p, n).T * (A_proj * B_proj)
            
            if not np.allclose(C_calc, C_expected, atol=1e-9):
                return False
        return True

    if not check_bilinear(U, V, W):
        # Fallback to schoolbook
        R_school = n * m * p
        U_school = np.zeros((n * m, R_school))
        V_school = np.zeros((m * p, R_school))
        W_school = np.zeros((p * n, R_school))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Try to reduce rank by removing zero columns or merging
    # For now, return the Strassen solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
