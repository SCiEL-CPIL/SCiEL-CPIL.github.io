import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Construct a rank 30 decomposition:
    # Rank 27 for 3x3x3 part + Rank 3 for 3x3x1 part
    rank_total = 30
    U = np.zeros((n*m, rank_total), dtype=int)
    V = np.zeros((m*p, rank_total), dtype=int)
    W = np.zeros((p*n, rank_total), dtype=int)
    
    col = 0
    # 3x3x3 part (k=0,1,2)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 4 + k
                c = k * 3 + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    # 3x3x1 part (k=3)
    # C[i,3] = sum_j A[i,j] B[j,3]
    # Terms for each j
    for j in range(3):
        b = j * 4 + 3
        for i in range(3):
            a = i * 3 + j
            c = 3 * 3 + i
            U[a, col] = 1
            V[b, col] = 1
            W[c, col] = 1
        col += 1
        
    # Verify
    T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
    if not np.allclose(T_decomp, T, atol=1e-10):
        # Fallback to trivial 36
        rank_fb = 36
        U_fb = np.zeros((n*m, rank_fb), dtype=int)
        V_fb = np.zeros((m*p, rank_fb), dtype=int)
        W_fb = np.zeros((p*n, rank_fb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
