"""
5x5 matrix multiplication - schoolbook rank 125.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = np.zeros(n*m)
                V = np.zeros(m*p)
                W = np.zeros(p*n)
                
                U[i*m + j] = 1.0
                V[j*p + k] = 1.0
                W[k*n + i] = 1.0
                
                U_list.append(U)
                V_list.append(V)
                W_list.append(W)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
