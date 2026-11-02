import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Use NumPy for efficient construction of the identity-like structure
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # The trivial decomposition corresponds to the tensor product of identities.
    # We can construct the matrices by placing 1s at the correct positions.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists to match the expected output format exactly
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
