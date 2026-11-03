import numpy as np
import time

def strassen_2x2_factors():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def build_strassen_factors(size):
        if time.time() > deadline:
            return None
        if size == 2:
            return strassen_2x2_factors()
        
        half = size // 2
        U_sub, V_sub, W_sub = build_strassen_factors(half)
        if U_sub is None:
            return None
            
        U_new = np.zeros((size*size, 49))
        V_new = np.zeros((size*size, 49))
        W_new = np.zeros((size*size, 49))
        
        a_coeffs = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        
        b_coeffs = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[0, -1], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        
        w_coeffs = [
            [[1, 0], [0, 1]],
            [[0, 0], [0, -1]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
            [[-1, 1], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        for k in range(7):
            for c in range(7):
                col_idx = k * 7 + c
                for sub_row in range(half*half):
                    sub_i = sub_row // half
                    sub_j = sub_row % half
                    
                    coeff_u = U_sub[sub_row, c]
                    if coeff_u != 0:
                        for bi in range(2):
                            for bj in range(2):
                                a_coeff = a_coeffs[k][bi][bj]
                                if a_coeff != 0:
                                    row_idx = (bi * half + sub_i) * size + (bj * half + sub_j)
                                    U_new[row_idx, col_idx] += a_coeff * coeff_u
                    
                    coeff_v = V_sub[sub_row, c]
                    if coeff_v != 0:
                        for bi in range(2):
                            for bj in range(2):
                                b_coeff = b_coeffs[k][bi][bj]
                                if b_coeff != 0:
                                    row_idx = (bi * half + sub_i) * size + (bj * half + sub_j)
                                    V_new[row_idx, col_idx] += b_coeff * coeff_v
                    
                    coeff_w = W_sub[sub_row, c]
                    if coeff_w != 0:
                        for ci in range(2):
                            for cj in range(2):
                                w_coeff = w_coeffs[k][ci][cj]
                                if w_coeff != 0:
                                    row_idx = (ci * half + sub_i) * size + (cj * half + sub_j)
                                    W_new[row_idx, col_idx] += w_coeff * coeff_w
        
        return U_new, V_new, W_new
    
    U_444, V_444, W_444 = build_strassen_factors(4)
    
    if U_444 is None:
        # Fallback to schoolbook if deadline exceeded
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U[:, 0:49] = U_444
    U[:, 49:98] = U_444
    
    V_444_perm = np.zeros((32, 49))
    for j in range(4):
        for k in range(4):
            V_444_perm[j*8 + k, :] = V_444[j*4 + k, :]
    V[0:32, 0:49] = V_444_perm
    
    V_444_perm2 = np.zeros((32, 49))
    for j in range(4):
        for k in range(4):
            V_444_perm2[j*8 + (k+4), :] = V_444[j*4 + k, :]
    V[0:32, 49:98] = V_444_perm2
    
    W[0:16, 0:49] = W_444
    W[16:32, 49:98] = W_444
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
