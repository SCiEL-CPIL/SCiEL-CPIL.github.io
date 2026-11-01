"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 11 = 44, using rank-11 decomposition for each 2x3 x 3x2 block
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Rank-11 decomposition for 2x3 x 3x2 matrix multiplication
    # A is 2x3, B is 3x2, C = A*B is 2x2
    # Using the known minimal decomposition
    
    # The 11 rank-1 terms for 2x3 x 3x2:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A01+A11) * B01  
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    # M8 = A12 * B00
    # M9 = A12 * B01
    # M10 = A02 * B10
    # M11 = A02 * B11
    
    # Wait, let me use the standard rank-11 for 2x3 x 3x2
    # Actually, let me use the same structure as the current program but verify
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] = 1
        U[4][col] = 1  # A[1,1] -> index 1*3+1=4
        V[k0][col] = 1   # B[0,k0]
        V[p + k1][col] = 1  # B[1,k1]
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3][col] = 1   # A[1,0]
        U[4][col] = 1   # A[1,1]
        V[k0][col] = 1   # B[0,k0]
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][col] = 1
        V[k1][col] = 1   # B[0,k1]
        V[p + k1][col] = -1  # B[1,k1]
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[4][col] = 1   # A[1,1]
        V[p + k0][col] = 1  # B[1,k0]
        V[k0][col] = -1   # B[0,k0]
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][col] = 1
        U[1][col] = 1   # A[0,1]
        V[p + k1][col] = 1  # B[1,k1]
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3][col] = 1   # A[1,0]
        U[0][col] = -1
        V[k0][col] = 1   # B[0,k0]
        V[k1][col] = 1   # B[0,k1]
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] = 1   # A[0,1]
        U[4][col] = -1  # A[1,1]
        V[p + k0][col] = 1  # B[1,k0]
        V[p + k1][col] = 1  # B[1,k1]
        W[k0 * n + 0][col] = 1
        col += 1
        
        # M8 = A02 * B2k0
        U[2][col] = 1   # A[0,2]
        V[2 * p + k0][col] = 1  # B[2,k0]
        W[k0 * n + 0][col] = 1
        col += 1
        
        # M9 = A02 * B2k1
        U[2][col] = 1   # A[0,2]
        V[2 * p + k1][col] = 1  # B[2,k1]
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M10 = A12 * B2k0
        U[5][col] = 1   # A[1,2]
        V[2 * p + k0][col] = 1  # B[2,k0]
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M11 = A12 * B2k1
        U[5][col] = 1   # A[1,2]
        V[2 * p + k1][col] = 1  # B[2,k1]
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
