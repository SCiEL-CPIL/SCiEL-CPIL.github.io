"""
Initial program for the suite task.
Implements the schoolbook decomposition for matrix multiplication 2x3 by 3x7.
"""

import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    Entries are integers.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for simplicity and exact integer arithmetic
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Schoolbook decomposition
    # Each term corresponds to (i, j, k)
    # U[i*m + j, column] = 1
    # V[j*p + k, column] = 1
    # W[k*n + i, column] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple validation to ensure the program runs
    res = construct()
    print("Constructed matrices.")
    print("Rank:", len(res['U'][0]))
