import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140  # Leave 10s margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Return schoolbook as fallback if we're running out of time
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2x2
    # We use the standard 7-term decomposition.
    # u, v are vectors in R^4 (flattened 2x2 matrices)
    # w is a vector in R^4 (flattened 2x2 output matrix)
    
    def get_strassen_terms():
        # Indices: 00, 01, 10, 11
        # M1 = (A00 + A11) * (B00 + B11)  -> C00 += M1, C11 += M1
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1]
        
        # M2 = (A10 + A11) * B00           -> C10 -= M2, C11 -= M2
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, -1, -1]
        
        # M3 = A00 * (B10 - B11)           -> C01 += M3, C11 += M3
        u3 = [1, 0, 0, 0]
        v3 = [0, 0, 1, -1]
        w3 = [0, 1, 0, 1]
        
        # M4 = A11 * (B01 - B00)           -> C00 += M4, C10 += M4
        u4 = [0, 0, 0, 1]
        v4 = [0, 1, -1, 0]
        w4 = [1, 0, 1, 0]
        
        # M5 = (A00 + A01) * B11           -> C00 -= M5, C01 += M5
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 1, 0, 0]
        
        # M6 = (A10 - A00) * (B00 + B01)   -> C11 += M6
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1]
        
        # M7 = (A01 - A11) * (B10 + B11)   -> C00 += M7, C11 -= M7
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, -1]
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
    
    terms = get_strassen_terms()
    
    # Verify Strassen locally on 2x2
    def verify_strassen():
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C_ref = A @ B
        C_calc = np.zeros((2, 2))
        for u, v, w in terms:
            u_a = np.array(u).reshape(2, 2)
            v_b = np.array(v).reshape(2, 2)
            w_c = np.array(w).reshape(2, 2)
            scalar_u = np.sum(u_a * A)
            scalar_v = np.sum(v_b * B)
            C_calc += scalar_u * scalar_v * w_c
        return np.allclose(C_calc, C_ref)
    
    if not verify_strassen():
        # Fallback to schoolbook rank 64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    # Proceed with rank 56
    # Structure:
    # A is 2x4. Split into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3).
    # B is 4x8. Split into 4 column blocks B_0, B_1, B_2, B_3, each 4x2.
    # Each B_b is split into B_b_T (2x2, rows 0,1) and B_b_B (2x2, rows 2,3).
    
    # C is 2x8. Split into 4 column blocks C_0, C_1, C_2, C_3, each 2x2.
    # C_b = A_L @ B_b_T + A_R @ B_b_B.
    
    # We use 7 Strassen terms for A_L @ B_b_T and 7 for A_R @ B_b_B.
    # Total 14 terms per column block.
    # 4 column blocks * 14 = 56 terms.
    
    target_rank = 56
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    current_col = 0
    for b in range(4):
        # Left part: A_L (cols 0,1) * B_b_T (rows 0,1, cols 2b, 2b+1)
        for u, v, w in terms:
            # u: 2x2 local A_L -> Global A cols 0,1
            for i in range(2):
                for j in range(2):
                    if u[i*2+j] != 0:
                        U[i * m + j][current_col] += u[i*2+j]
                        
            # v: 2x2 local B_b_T -> Global B rows 0,1, cols 2b, 2b+1
            for j in range(2):
                for k in range(2):
                    if v[j*2+k] != 0:
                        V[j * p + (2*b + k)][current_col] += v[j*2+k]
                        
            # w: 2x2 local C_b -> Global C cols 2b, 2b+1
            # Output index in W is (k*n + i) where k is col index in C, i is row index in C.
            for i in range(2):
                for k in range(2):
                    if w[i*2+k] != 0:
                        W[(2*b + k) * n + i][current_col] += w[i*2+k]
                        
            current_col += 1
            
        # Right part: A_R (cols 2,3) * B_b_B (rows 2,3, cols 2b, 2b+1)
        for u, v, w in terms:
            # u: 2x2 local A_R -> Global A cols 2,3
            for i in range(2):
                for j in range(2):
                    if u[i*2+j] != 0:
                        U[i * m + (2 + j)][current_col] += u[i*2+j]
                        
            # v: 2x2 local B_b_B -> Global B rows 2,3, cols 2b, 2b+1
            for j in range(2):
                for k in range(2):
                    if v[j*2+k] != 0:
                        V[(2 + j) * p + (2*b + k)][current_col] += v[j*2+k]
                        
            # w: 2x2 local C_b -> Global C cols 2b, 2b+1
            for i in range(2):
                for k in range(2):
                    if w[i*2+k] != 0:
                        W[(2*b + k) * n + i][current_col] += w[i*2+k]
                        
            current_col += 1
            
    # Final Verification
    def verify_full():
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_ref = A @ B
        
        U_mat = np.array(U).T # rank x n*m
        V_mat = np.array(V).T # rank x m*p
        W_mat = np.array(W).T # rank x p*n
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # proj_U = U_mat @ A_flat  (rank vector)
        # proj_V = V_mat @ B_flat  (rank vector)
        # coeff = proj_U * proj_V
        # C_flat = coeff @ W_mat
        
        proj_U = U_mat @ A_flat
        proj_V = V_mat @ B_flat
        coeff = proj_U * proj_V
        C_flat = coeff @ W_mat
        
        return np.allclose(C_flat.reshape(n, p), C_ref)
        
    if not verify_full():
        # Fallback to schoolbook rank 64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
