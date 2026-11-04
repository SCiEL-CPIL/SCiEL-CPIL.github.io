"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    rank_schoolbook = n * m * p
    # We replace 8 terms in the sub-block (0,1)x(0,1)x(0,1) with 7 Strassen terms
    rank_strassen = 7
    removed_terms = 8
    rank = rank_schoolbook - removed_terms + rank_strassen
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for the sub-block
    # i in {0, 1}, j in {0, 1}, k in {0, 1}
    # U indices: i*m + j -> 0, 1, 5, 6
    # V indices: j*p + k -> 0, 1, 6, 7
    # W indices: k*n + i -> 0, 1, 4, 5
    
    col = 0
    
    # Strassen Part
    # P1 = (A00 + A11)(B00 + B11)
    U[0][col] = 1; U[6][col] = 1
    V[0][col] = 1; V[7][col] = 1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # P2 = (A10 + A11)B00
    U[5][col] = 1; U[6][col] = 1
    V[0][col] = 1
    W[4][col] = 1; W[5][col] = -1
    col += 1
    
    # P3 = A00(B01 - B11)
    U[0][col] = 1
    V[1][col] = 1; V[7][col] = -1
    W[1][col] = 1; W[5][col] = 1
    col += 1
    
    # P4 = A11(B10 - B00)
    U[6][col] = 1
    V[6][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # P5 = (A00 + A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[5][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[1][col] = 1; U[6][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # Schoolbook Part for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the sub-block handled by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
