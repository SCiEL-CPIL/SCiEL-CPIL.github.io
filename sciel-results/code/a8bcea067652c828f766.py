import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for k=0, 1
    # M1 = (A00+A11)(B00+B01)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1 # W[k*n+i]
    # W indices: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1 # C10
    
    # M3 = A00(B11-B10)
    U[0][2] = 1
    V[1][2] = 1; V[0][2] = -1
    W[0][2] = 1; W[2][2] = 1 # C00, C01
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[3][3] = 1 # C10, C11
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[1][4] = 1
    W[0][4] = 1; W[2][4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1; W[3][5] = 1 # C10, C11
    
    # M7 = (A01-A11)(B11+B10)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[0][6] = 1
    W[0][6] = 1; W[2][6] = 1 # C00, C01
    
    # For k=2..6
    col = 7
    for k in range(2, p):
        # M_{3k-2} = A00 B0k
        U[0][col] = 1
        V[k][col] = 1
        W[k*n+0][col] = 1
        
        # M_{3k-1} = A01 B1k
        U[1][col+1] = 1
        V[p+k][col+1] = 1
        W[k*n+0][col+1] = 1
        
        # M_{3k} = A10 B0k + A11 B1k
        U[2][col+2] = 1
        U[3][col+2] = 1
        V[k][col+2] = 1
        V[p+k][col+2] = 1
        W[k*n+1][col+2] = 1
        
        col += 3
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
