import time
import sys

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication of dimensions 
    n x m by m x p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    rank = n*m*p
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # Using lists of lists for simplicity and compatibility with the expected output format
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the decomposition
    # The tensor for matrix multiplication C = A * B is given by C[i, k] = sum_j A[i, j] B[j, k]
    # In the tensor decomposition context:
    # U[i*m + j, c] corresponds to A[i, j]
    # V[j*p + k, c] corresponds to B[j, k]
    # W[k*n + i, c] corresponds to C[i, k] (note the trace convention W[k*n+i, r])
    # Wait, the problem statement says: "The output factor follows the official trace convention W[k*n+i,r]."
    # Standard schoolbook:
    # For each i, j, k:
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
