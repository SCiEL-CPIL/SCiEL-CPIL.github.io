"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process 5 pairs of columns (0..9)
    for k in range(5):
        # b11 = B[0, 2k] -> idx 2k
        # b12 = B[0, 2k+1] -> idx 2k+1
        # b21 = B[1, 2k] -> idx 11+2k
        # b22 = B[1, 2k+1] -> idx 11+2k+1
        
        # M1: (A00 + A11) * (B0,2k + B1,2k+1)
        U[0][col] = 1
        U[3][col] = 1
        V[2*k][col] = 1
        V[11+2*k+1][col] = 1
        W[4*k][col] = 1      # C[2k, 0]
        W[4*k+3][col] = 1    # C[2k+1, 1]
        col += 1
        
        # M2: (A10 + A11) * B0,2k
        U[2][col] = 1
        U[3][col] = 1
        V[2*k][col] = 1
        W[4*k+1][col] = 1    # C[2k, 1]
        W[4*k+3][col] = -1   # C[2k+1, 1]
        col += 1
        
        # M3: A00 * (B0,2k+1 - B1,2k+1)
        U[0][col] = 1
        V[2*k+1][col] = 1
        V[11+2*k+1][col] = -1
        W[4*k+2][col] = 1    # C[2k+1, 0]
        W[4*k+3][col] = 1    # C[2k+1, 1]
        col += 1
        
        # M4: A11 * (B1,2k - B0,2k)
        U[3][col] = 1
        V[11+2*k][col] = 1
        V[2*k][col] = -1
        W[4*k][col] = 1      # C[2k, 0]
        W[4*k+1][col] = 1    # C[2k, 1]
        col += 1
        
        # M5: (A00 + A01) * B1,2k+1
        U[0][col] = 1
        U[1][col] = 1
        V[11+2*k+1][col] = 1
        W[4*k][col] = -1     # C[2k, 0]
        W[4*k+2][col] = 1    # C[2k+1, 0]
        col += 1
        
        # M6: (A10 - A00) * (B0,2k + B0,2k+1)
        U[2][col] = 1
        U[0][col] = -1
        V[2*k][col] = 1
        V[2*k+1][col] = 1
        W[4*k+3][col] = 1    # C[2k+1, 1]
        col += 1
        
        # M7: (A01 - A11) * (B1,2k + B1,2k+1)
        U[1][col] = 1
        U[3][col] = -1
        V[11+2*k][col] = 1
        V[11+2*k+1][col] = 1
        W[4*k][col] = 1      # C[2k, 0]
        col += 1
        
    # Process last column (index 10)
    # M8: A00 * B0,10
    U[0][col] = 1
    V[10][col] = 1
    W[20][col] = 1 # C[10, 0]
    col += 1
    
    # M9: A01 * B1,10
    U[1][col] = 1
    V[21][col] = 1
    W[20][col] = 1 # C[10, 0]
    col += 1
    
    # M10: A10 * B0,10
    U[2][col] = 1
    V[10][col] = 1
    W[21][col] = 1 # C[10, 1]
    col += 1
    
    # M11: A11 * B1,10
    U[3][col] = 1
    V[21][col] = 1
    W[21][col] = 1 # C[10, 1]
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
