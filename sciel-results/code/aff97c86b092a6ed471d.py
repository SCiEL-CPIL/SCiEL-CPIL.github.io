import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for j in range(m):
        for k_prime in range(p):
            r = j * p + k_prime
            # U is all 1s
            U[:, r] = 1
            # V[j*p + k, r] is 1 if k == k_prime
            V[j * p + k_prime, r] = 1
            # W[k*n + i, r] is 1 if k == k_prime (for all i)
            for i in range(n):
                W[k_prime * n + i, r] = 1
                
    return {"U": U, "V": V, "W": W}
