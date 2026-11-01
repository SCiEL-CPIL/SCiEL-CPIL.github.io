"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank 11 decomposition for 2x2 by 2x3 matrix multiplication
    # Uses Strassen for the first 2 columns (rank 7) and schoolbook for the 3rd column (rank 4)
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 x 2x2 part (indices k=0,1)
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1 # c00, c11
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1 # c10
    
    # M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = -1; W[1][2] = 1 # c01, c11
    
    # M4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1 # c00, c10
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = -1 # c00, c11
    
    # M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1 # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # c00
    
    # Schoolbook for 3rd column (k=2)
    # C02 = a00*b02 + a01*b12
    # C12 = a10*b02 + a11*b12
    
    # M8 = a00*b02
    U[0][7] = 1
    V[2][7] = 1 # b02 is index 0*3+2 = 2
    W[2][7] = 1 # c02 is index 2*2+0 = 4? No, W index is k*n+i. k=2, i=0 -> 4.
    # Wait, W shape is (p*n, rank). p=3, n=2. Size 6.
    # Index k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    
    # Correcting W indices for Strassen:
    # W[0] is c00 (k=0,i=0)
    # W[1] is c10 (k=0,i=1) -> No, c10 is i=1, k=0. Index 1.
    # W[2] is c01 (k=1,i=0). Index 2.
    # W[3] is c11 (k=1,i=1). Index 3.
    # W[4] is c02 (k=2,i=0). Index 4.
    # W[5] is c12 (k=2,i=1). Index 5.
    
    # Re-evaluating Strassen W assignments:
    # M1: c00, c11. W[0]=1, W[3]=1.
    # M2: c10. W[1]=1.
    # M3: c01, c11. c01 is W[2], c11 is W[3].
    # M4: c00, c10. c00 is W[0], c10 is W[1].
    # M5: c00, c11. c00 is W[0], c11 is W[3].
    # M6: c11. W[3]=1.
    # M7: c00. W[0]=1.
    
    # Let's rewrite the whole thing cleanly.
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = -1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    # M8 = a00*b02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # M9 = a01*b12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # M10 = a10*b02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # M11 = a11*b12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
