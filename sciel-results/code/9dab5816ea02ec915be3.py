import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A is n x m, index i*m + j
                # B is m x p, index j*p + k
                # C is n x p, index i*p + k
                # But W index is k*n + i (from previous analysis)
                # Let's re-verify W index.
                # Prompt: W[k*n+i, r]
                # C is n x p. C[i, k].
                # If W index is k*n + i, then W[k*n+i] corresponds to C[i, k].
                # So for term (i, j, k):
                # U[i*m + j]
                # V[j*p + k]
                # W[k*n + i]
                
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
