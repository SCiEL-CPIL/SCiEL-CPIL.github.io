"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for first 2 columns (k=0,1)
    # m1 = (A00 + A11)(B00 + B11)
    # U: A00, A11
    U[0][0] = 1; U[3][0] = 1
    # V: B00, B11
    V[0][0] = 1; V[4][0] = 1
    # W: C00 (+1), C11 (+1)
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # W: C10 (+1)
    W[1][1] = 1
    
    # m3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    # W: C01 (+1)
    W[2][2] = 1
    
    # m4 = A11(B10 - B00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    # W: C00 (+1), C10 (+1)
    W[0][3] = 1; W[1][3] = 1
    
    # m5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    # W: C00 (-1), C01 (+1)
    W[0][4] = -1; W[2][4] = 1
    
    # m6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # W: C11 (+1)
    W[3][5] = 1
    
    # m7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    # W: C00 (+1)
    W[0][6] = 1
    
    # Schoolbook for column 2 (k=2)
    # m8 = A00 B02
    U[0][7] = 1
    V[2][7] = 1
    # W: C02 (+1)
    W[4][7] = 1
    
    # m9 = A01 B12
    U[1][8] = 1
    V[5][8] = 1
    # W: C02 (+1)
    W[4][8] = 1
    
    # m10 = A10 B02
    U[2][9] = 1
    V[2][9] = 1
    # W: C12 (+1)
    W[5][9] = 1
    
    # m11 = A11 B12
    U[3][10] = 1
    V[5][10] = 1
    # W: C12 (+1)
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
