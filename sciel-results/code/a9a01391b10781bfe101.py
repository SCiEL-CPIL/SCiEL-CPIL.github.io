"""Rewritten program for matrix multiplication 3,3,7."""
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    # Convert to lists as required by interface (if strict, but usually numpy arrays are accepted or converted)
    # The interface says 'U', 'V', 'W' with integer/half-integer entries.
    # Returning numpy arrays might be fine, but let's return lists to be safe if the evaluator expects lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
