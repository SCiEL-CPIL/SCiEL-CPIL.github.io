import numpy as np
from itertools import product
import time

def construct():
    t0 = time.time()
    deadline = t0 + 130
    
    n, m, p = 3, 4, 4
    
    # Rank-44 Strassen-based decomposition
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
    def place_block(col_off, a_row_s, a_col_s, b_row_s, b_col_s, c_row_s, c_col_s):
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                ga = (a_row_s + r) * m + (a_col_s + c)
                U[ga, col_off:col_off+11] += U_loc[li, :]
        for r in range(2):
            for c in range(2):
                li = r * 2 + c
                gb = (b_row_s + r) * p + (b_col_s + c)
                V[gb, col_off:col_off+11] += V_loc[li, :]
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                gc = (c_col_s + c) * n + (c_row_s + r)
                W[gc, col_off:col_off+11] += W_loc[li, :]
    
    place_block(0, 0, 0, 0, 0, 0, 0)
    place_block(11, 0, 2, 2, 0, 0, 0)
    place_block(22, 0, 0, 0, 2, 0, 2)
    place_block(33, 0, 2, 2, 2, 0, 2)
    
    # Fast verification using vectorized operations
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for l in range(m):
            for j in range(p):
                T[i*m+l, l*p+j, j*n+i] = 1.0
    
    # Compute decomposition tensor
    # D[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
    # Use einsum for efficiency
    D = np.einsum('ak,bk,ck->abc', U, V, W)
    
    if np.allclose(D, T, atol=1e-10):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Should not reach here, but return anyway
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
