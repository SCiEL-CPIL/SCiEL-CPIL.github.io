"""
Matrix multiplication tensor decomposition for 3x3 x 3x4.
Uses a splitting strategy to achieve a lower rank than schoolbook (36).
Split M=3 into 1+2.
Part 1: 3x1x4 (Rank 1)
Part 2: 3x2x4 (Split P=4 into 2+2 -> 2 * Rank(3x2x2))
Rank(3x2x2) is 2? No, Rank(3x2x2) is 12 in schoolbook.
Wait, if Rank(3x2x2) is 12, then Part 2 is 24.
Total 25.
"""
import time
import random
import numpy as np

N, M, P = 3, 3, 4
DIMENSIONS = [N, M, P]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the factorization is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    R = n * m * p
    U = [[0.0] * R for _ in range(n*m)]
    V = [[0.0] * R for _ in range(m*p)]
    W = [[0.0] * R for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return U, V, W

def _construct_split(n, m, p):
    """
    Construct a decomposition by splitting M into 1 and 2.
    Part 1: 3x1x4 (Rank 1)
    Part 2: 3x2x4 (Split P into 2+2, each 3x2x2 Rank 12)
    Total Rank: 1 + 12 + 12 = 25.
    """
    # Part 1: j=0 (3x1x4)
    # U[i*3+0, 0] = 1, V[0*4+k, 0] = 1, W[k*3+i, 0] = 1
    # Part 2: j=1,2 (3x2x4)
    # Split P=4 into k=0,1 and k=2,3.
    # Subpart 2a: k=0,1 (3x2x2)
    # Subpart 2b: k=2,3 (3x2x2)
    
    rank = 25
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    c = 0
    # Part 1: j=0
    for i in range(n):
        for k in range(p):
            U[i*m+0][c] = 1.0
            V[0*p+k][c] = 1.0
            W[k*n+i][c] = 1.0
            c += 1
            
    # Part 2: j=1,2
    # Subpart 2a: k=0,1
    for i in range(n):
        for j in [1, 2]:
            for k in [0, 1]:
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
                
    # Subpart 2b: k=2,3
    for i in range(n):
        for j in [1, 2]:
            for k in [2, 3]:
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
                
    return U, V, W

def construct():
    n, m, p = N, M, P
    
    # Try the split decomposition
    U, V, W = _construct_split(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U_school, V_school, W_school = _schoolbook(n, m, p)
        return {"U": U_school, "V": V_school, "W": W_school}
