"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    
    # U: (n*m) x rank = 4 x 11
    # V: (m*p) x rank = 6 x 11
    # W: (p*n) x rank = 6 x 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U row: 2*i + j  (i=0,1; j=0,1) -> a_ij
    # V row: 3*j + k  (j=0,1; k=0,1,2) -> b_jk
    # W row: 2*k + i  (k=0,1,2; i=0,1) -> c_ki
    
    # Col 0: M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Col 1: M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Col 2: M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Col 3: M4 = a11(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = -1
    
    # Col 4: M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Col 5: M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Col 6: M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Col 7: M8 = a00*b02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Col 8: M9 = a01*b12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Col 9: M10 = a10*b02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Col 10: M11 = a11*b12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
