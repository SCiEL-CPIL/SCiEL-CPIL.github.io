import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Constructs the factor matrices for the tensor decomposition of matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # Trivial rank is n * m * p
    R = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    # Fill in the trivial decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For the trivial decomposition, each term is a basis tensor.
    # Index r corresponds to the triplet (i, j, k).
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() > deadline:
                    # If we run out of time, return what we have (which is incomplete/invalid)
                    # But for trivial, it's instant.
                    pass
                    
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][idx] = 1.0
                V[v_idx][idx] = 1.0
                W[w_idx][idx] = 1.0
                idx += 1
                
    return {"U": U, "V": V, "W": W}
