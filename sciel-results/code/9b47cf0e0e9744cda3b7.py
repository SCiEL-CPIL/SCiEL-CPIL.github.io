import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Exploit linearity in the output dimension.
    # Decompose the multiplication into p independent (n, m, 1) tensor instances.
    # The (n, m, 1) tensor corresponds to multiplying an n x m matrix by an m x 1 vector.
    # The rank of the (n, m, 1) tensor is n * m.
    # Therefore, the total rank is p * (n * m).
    
    rank = p * n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over the output columns k, treating each as an independent (n, m, 1) problem
    for k in range(p):
        # For each output column k, we sum over i and j
        for i in range(n):
            for j in range(m):
                # Index in U: row i*m + j
                U[i * m + j][col] = 1
                # Index in V: row j*p + k
                V[j * p + k][col] = 1
                # Index in W: row k*n + i (trace convention)
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
