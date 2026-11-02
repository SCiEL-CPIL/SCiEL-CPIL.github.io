import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    rank = n * m * p  # 45

    # Initialize U, V, W with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i  # W[k*n+i, r] convention
                U[u_idx, column] = 1
                V[v_idx, column] = 1
                W[w_idx, column] = 1
                column += 1

    # Convert to lists if the evaluator strictly expects lists, 
    # but numpy arrays are usually accepted and more efficient.
    # The prompt says "factor matrices of shapes...", numpy arrays are matrices.
    # I will return numpy arrays.
    
    return {"U": U, "V": V, "W": W}
