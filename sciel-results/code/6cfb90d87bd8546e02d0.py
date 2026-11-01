"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 9
    
    # U: (n*m, rank) = (4, 9)
    # V: (m*p, rank) = (6, 9)
    # W: (p*n, rank) = (6, 9)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set column
    def set_col(idx, u, v, w):
        for i in range(n * m):
            U[i][idx] = u[i]
        for i in range(m * p):
            V[i][idx] = v[i]
        for i in range(p * n):
            W[i][idx] = w[i]

    # Strassen decomposition for first 2 columns (M_2,2,2)
    # Indices for A: 0:a11, 1:a12, 2:a21, 3:a22
    # Indices for B: 0:b11, 1:b12, 2:b13, 3:b21, 4:b22, 5:b23
    # Indices for C: 0:c11, 1:c21, 2:c12, 3:c22, 4:c13, 5:c23
    
    # P1
    set_col(0, 
            [1, 0, 0, 1], 
            [1, 0, 0, 0, 1, 0], 
            [1, 0, 0, 1, 0, 0])
            
    # P2
    set_col(1, 
            [0, 0, 1, 1], 
            [1, 0, 0, 0, 0, 0], 
            [0, 1, 0, -1, 0, 0])
            
    # P3
    set_col(2, 
            [1, 0, 0, 0], 
            [0, 1, 0, 0, -1, 0], 
            [0, 0, 1, 1, 0, 0])
            
    # P4
    set_col(3, 
            [0, 0, 0, 1], 
            [-1, 0, 0, 1, 0, 0], 
            [1, 1, 0, 0, 0, 0])
            
    # P5
    set_col(4, 
            [1, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0], 
            [-1, 0, 1, 0, 0, 0])
            
    # P6
    set_col(5, 
            [-1, 0, 1, 0], 
            [1, 1, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0])
            
    # P7
    set_col(6, 
            [0, 1, 0, -1], 
            [0, 0, 0, 1, 1, 0], 
            [1, 0, 0, 0, 0, 0])
            
    # Rank 2 decomposition for M_2,2,1 (3rd column)
    # Term 1: b13 * (a11*c13 + a21*c23)
    set_col(7, 
            [1, 0, 1, 0], 
            [0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 0, 1, 1])
            
    # Term 2: b23 * (a12*c13 + a22*c23)
    set_col(8, 
            [0, 1, 0, 1], 
            [0, 0, 0, 0, 0, 1], 
            [0, 0, 0, 0, 1, 1])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
