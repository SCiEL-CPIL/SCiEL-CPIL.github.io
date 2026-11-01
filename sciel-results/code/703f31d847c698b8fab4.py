"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 (k=0, i=0)
    W[3][0] = 1  # C11 (k=1, i=1)
    
    # M2 = (A10 + A11)B00 -> C10, -C11
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 (k=0, i=1)
    W[3][1] = -1 # C11 (k=1, i=1)
    
    # M3 = A00(B01 - B11) -> C01, C11
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    W[1][2] = 1  # C01 (k=1, i=0)
    W[3][2] = 1  # C11 (k=1, i=1)
    
    # M4 = A11(B10 - B00) -> C00, C10
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[0][3] = 1  # C00 (k=0, i=0)
    W[2][3] = 1  # C10 (k=0, i=1)
    
    # M5 = (A00 + A01)B11 -> -C00, C01
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1 # C00 (k=0, i=0)
    W[1][4] = 1  # C01 (k=1, i=0)
    
    # M6 = (A10 - A00)(B00 + B01) -> C11
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11 (k=1, i=1)
    
    # M7 = (A01 - A11)(B10 + B11) -> C00
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00 (k=0, i=0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
