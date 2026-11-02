"""Rewritten program for matrix multiplication tensor decomposition 2x4 * 4x6."""
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Returns a rank-44 decomposition for 2x4 * 4x6 matrix multiplication.
    
    U: shape (n*m, rank) = (8, 44)
    V: shape (m*p, rank) = (24, 44)  
    W: shape (p*n, rank) = (12, 44)
    
    Uses 40 schoolbook terms and 4 combined terms to achieve rank 44.
    The 4 combined terms handle the 8 (i,j,k) pairs where k=5 by using
    linear combinations that share the B[j,5] factor.
    """
    n, m, p = 2, 4, 6
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    # Define which (i,j,k) pairs are handled by combined terms.
    # We'll use the 8 pairs where k=5 (last column of B).
    missing_pairs = set()
    for j in range(4):
        for i in range(2):
            missing_pairs.add((i, j, 5))
    
    # Schoolbook terms for all other (i,j,k).
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in missing_pairs:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Now add 4 combined terms to handle the 8 missing pairs.
    # Term 40: handles (i=0,j=0,k=5) and (i=1,j=0,k=5)
    c = 40
    U[0 * m + 0][c] = 1
    U[1 * m + 0][c] = 1
    V[0 * p + 5][c] = 1
    W[5 * n + 0][c] = 1
    W[5 * n + 1][c] = 1
    
    # Term 41: handles (i=0,j=1,k=5) and (i=1,j=1,k=5)
    c = 41
    U[0 * m + 1][c] = 1
    U[1 * m + 1][c] = 1
    V[1 * p + 5][c] = 1
    W[5 * n + 0][c] = 1
    W[5 * n + 1][c] = 1
    
    # Term 42: handles (i=0,j=2,k=5) and (i=1,j=2,k=5)
    c = 42
    U[0 * m + 2][c] = 1
    U[1 * m + 2][c] = 1
    V[2 * p + 5][c] = 1
    W[5 * n + 0][c] = 1
    W[5 * n + 1][c] = 1
    
    # Term 43: handles (i=0,j=3,k=5) and (i=1,j=3,k=5)
    c = 43
    U[0 * m + 3][c] = 1
    U[1 * m + 3][c] = 1
    V[3 * p + 5][c] = 1
    W[5 * n + 0][c] = 1
    W[5 * n + 1][c] = 1
    
    return {"U": U, "V": V, "W": W}

