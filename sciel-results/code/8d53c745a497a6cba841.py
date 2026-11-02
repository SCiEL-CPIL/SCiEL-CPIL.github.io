"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2 (k=0,1)
    # Indices for U: i*m+j
    # Indices for V: j*p+k
    # Indices for W: k*n+i
    
    # r=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # r=3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # r=4
    U[3][3] = 1
    V[0][3] = -1; V[2][3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # r=5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # r=6
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1
    
    # New terms for k=2 (r=8,9,10)
    # We use 3 terms to satisfy the 4 equations for k=2
    # r=8
    U[0][7] = 1; U[1][7] = 1; U[2][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[5][7] = 1
    W[4][7] = 1; W[5][7] = 1
    
    # r=9
    U[0][8] = 1; U[1][8] = 1; U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1; V[5][8] = 1
    W[4][8] = 1; W[5][8] = -1
    
    # r=10
    U[0][9] = 1; U[1][9] = 1; U[2][9] = 1; U[3][9] = 1
    V[2][9] = 1; V[5][9] = 1
    W[4][9] = -1; W[5][9] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
