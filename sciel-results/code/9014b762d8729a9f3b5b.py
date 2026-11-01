"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=3
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (6, rank)
    # W: (p*n, rank) = (6, rank)
    
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (U): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (V): 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # Indices for C (W): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    
    # Strassen terms for k=0,1 (B indices 0,1,3,4; C indices 0,1,2,3)
    
    # 1. M1 = (a+d)(e+g)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # 2. M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # 3. M3 = a(f-g)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # 4. M4 = d(i-g)
    U[3][3] = 1
    V[3][3] = -1; V[4][3] = 1
    W[0][3] = 1; W[2][3] = 1
    
    # 5. M5 = (a+b)i
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # 6. M6 = (b+c)(f-e)
    U[1][5] = 1; U[2][5] = 1
    V[0][5] = -1; V[1][5] = 1
    W[3][5] = 1
    
    # 7. M7 = (a-d)(e+f)
    U[0][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    # Schoolbook terms for k=2 (B indices 2,5; C indices 4,5)
    
    # 8. a * B[0,2] -> C[2,0]
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # 9. b * B[0,2] -> C[2,1]
    U[1][8] = 1
    V[2][8] = 1
    W[5][8] = 1
    
    # 10. c * B[1,2] -> C[2,0]
    U[2][9] = 1
    V[5][9] = 1
    W[4][9] = 1
    
    # 11. d * B[1,2] -> C[2,1]
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
