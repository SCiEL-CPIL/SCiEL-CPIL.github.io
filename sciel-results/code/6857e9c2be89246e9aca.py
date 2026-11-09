import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank for n x m x p is n*m*p
    # We can split p into p1=3 and p2=1
    # Rank for 3x3x3 is 27 (schoolbook)
    # Rank for 3x3x1 is 3 (schoolbook)
    # Total rank 30
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 3x3x3 multiplication (k=0,1,2)
    for i in range(n):
        for j in range(m):
            for k in range(3): # k=0,1,2
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 3x3x1 multiplication (k=3)
    for i in range(n):
        for j in range(m):
            # k=3
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
