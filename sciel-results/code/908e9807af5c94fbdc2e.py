import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index into U: row is i*m + j
                U[i * m + j, col] = 1.0
                # Index into V: row is j*p + k
                V[j * p + k, col] = 1.0
                # Index into W: row is k*n + i
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists to match the expected interface if necessary, 
    # though numpy arrays are often accepted. The seed used lists of lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
