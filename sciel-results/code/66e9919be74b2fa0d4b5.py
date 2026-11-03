"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # Strassen Terms (1-7)
    # P1 = (A00+A11)(B00+B11)
    U[0 * m + 0][0] = 1
    U[1 * m + 1][0] = 1
    V[0 * p + 0][0] = 1
    V[1 * p + 1][0] = 1
    W[0 * n + 0][0] = 1 # C00
    W[1 * n + 1][0] = 1 # C11
    
    # P2 = (A10+A11)B00
    U[1 * m + 0][1] = 1
    U[1 * m + 1][1] = 1
    V[0 * p + 0][1] = 1
    W[0 * n + 1][1] = 1 # C10
    W[1 * n + 1][1] = -1 # C11
    
    # P3 = A00(B10-B11)
    U[0 * m + 0][2] = 1
    V[1 * p + 0][2] = 1
    V[1 * p + 1][2] = -1
    W[1 * n + 0][2] = 1 # C01
    W[1 * n + 1][2] = 1 # C11
    
    # P4 = A11(B01-B00)
    U[1 * m + 1][3] = 1
    V[0 * p + 1][3] = 1
    V[0 * p + 0][3] = -1
    W[0 * n + 0][3] = 1 # C00
    W[0 * n + 1][3] = 1 # C10
    
    # P5 = (A00+A01)B11
    U[0 * m + 0][4] = 1
    U[0 * m + 1][4] = 1
    V[1 * p + 1][4] = 1
    W[0 * n + 0][4] = -1 # C00
    W[1 * n + 0][4] = 1  # C01
    
    # P6 = (A10-A00)(B00+B01)
    U[1 * m + 0][5] = 1
    U[0 * m + 0][5] = -1
    V[0 * p + 0][5] = 1
    V[0 * p + 1][5] = 1
    W[1 * n + 1][5] = 1  # C11
    
    # P7 = (A01-A11)(B10+B11)
    U[0 * m + 1][6] = 1
    U[1 * m + 1][6] = -1
    V[1 * p + 0][6] = 1
    V[1 * p + 1][6] = 1
    W[0 * n + 0][6] = 1  # C00
    
    # Schoolbook Terms (8-17)
    # Term 8: A02 B20 -> C00
    U[0 * m + 2][7] = 1
    V[2 * p + 0][7] = 1
    W[0 * n + 0][7] = 1
    
    # Term 9: A02 B21 -> C01
    U[0 * m + 2][8] = 1
    V[2 * p + 1][8] = 1
    W[1 * n + 0][8] = 1
    
    # Term 10: A12 B20 -> C10
    U[1 * m + 2][9] = 1
    V[2 * p + 0][9] = 1
    W[0 * n + 1][9] = 1
    
    # Term 11: A12 B21 -> C11
    U[1 * m + 2][10] = 1
    V[2 * p + 1][10] = 1
    W[1 * n + 1][10] = 1
    
    # Term 12: A00 B02 -> C02
    U[0 * m + 0][11] = 1
    V[0 * p + 2][11] = 1
    W[2 * n + 0][11] = 1
    
    # Term 13: A01 B12 -> C02
    U[0 * m + 1][12] = 1
    V[1 * p + 2][12] = 1
    W[2 * n + 0][12] = 1
    
    # Term 14: A02 B22 -> C02
    U[0 * m + 2][13] = 1
    V[2 * p + 2][13] = 1
    W[2 * n + 0][13] = 1
    
    # Term 15: A10 B02 -> C12
    U[1 * m + 0][14] = 1
    V[0 * p + 2][14] = 1
    W[2 * n + 1][14] = 1
    
    # Term 16: A11 B12 -> C12
    U[1 * m + 1][15] = 1
    V[1 * p + 2][15] = 1
    W[2 * n + 1][15] = 1
    
    # Term 17: A12 B22 -> C12
    U[1 * m + 2][16] = 1
    V[2 * p + 2][16] = 1
    W[2 * n + 1][16] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
