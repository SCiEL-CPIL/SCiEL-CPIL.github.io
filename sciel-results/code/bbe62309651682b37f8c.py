import time
import numpy as np

# Deadline guard
_START = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 7-product decomposition for 2x2x2
    # U_coeffs[s][(i,j)] = coefficient of M_s in A[i,j]
    # V_coeffs[s][(j,k)] = coefficient of M_s in B[j,k]
    # W_coeffs[s][(i,k)] = coefficient of M_s in C[i,k]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},    # M1 = (A11+A22)(B11+B22)
        {(1, 0): 1, (1, 1): 1},    # M2 = (A21+A22)B11
        {(0, 0): 1},               # M3 = A11(B21-B11)
        {(1, 1): 1},               # M4 = A22(B12-B22)
        {(0, 0): 1, (0, 1): 1},    # M5 = (A11+A12)B22
        {(1, 0): 1, (0, 0): -1},   # M6 = (A21-A11)(B11+B12)
        {(0, 1): 1, (1, 1): -1}    # M7 = (A12-A22)(B21+B22)
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},    # M1
        {(0, 0): 1},               # M2
        {(0, 1): 1, (1, 1): -1},   # M3
        {(1, 0): 1, (0, 0): -1},   # M4
        {(1, 1): 1},               # M5
        {(0, 0): 1, (0, 1): 1},    # M6
        {(1, 0): 1, (1, 1): 1}     # M7
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},    # M1 -> C11, C22
        {(1, 0): 1, (1, 1): -1},   # M2 -> C21, -C22
        {(0, 1): 1, (1, 1): 1},    # M3 -> C12, C22
        {(0, 0): 1, (1, 0): 1},    # M4 -> C11, C21
        {(0, 0): -1, (0, 1): 1},   # M5 -> -C11, C12
        {(1, 1): 1},               # M6 -> C22
        {(0, 0): 1}                # M7 -> C11
    ]
    
    col = 0
    for g in range(2):  # 2 blocks in the column direction of A (j = 0,1 and 2,3)
        for h in range(3):  # 3 blocks in the row direction of B (k = 0,1; 2,3; 4,5)
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = g * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = g * 2 + j_loc
                    k = h * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = h * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
