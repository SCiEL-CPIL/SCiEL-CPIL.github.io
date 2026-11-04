import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms
    # P1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1 # C10
    
    # P3
    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[2][2] = 1 # C01
    
    # P4
    U[4][3] = 1
    V[5][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1 # C00, C10
    
    # P5
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = 1; W[2][4] = 1 # C00, C01
    
    # P6
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11
    
    # P7
    U[1][6] = 1; U[4][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = 1; W[3][6] = 1 # C00, C11
    
    # Schoolbook terms
    col = 7
    # j=2, k in 0,1; i in 0,1
    for i in range(2):
        for k in range(2):
            U[i*3 + 2][col] = 1
            V[2*5 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
            
    # j in 0,1,2; k in 2,3,4; i in 0,1
    for i in range(2):
        for j in range(3):
            for k in range(2, 5):
                U[i*3 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
