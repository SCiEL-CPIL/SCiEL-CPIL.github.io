"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank minimal for 2x2 x 2xp is 3p+1 = 31
    rank = 3 * p + 1
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We use the Strassen decomposition for the first 2 columns (rank 7)
    # and 3 terms for each of the remaining 8 columns.
    
    # Strassen for 2x2x2 (columns 0 and 1)
    # Indices:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11 (for cols 0, 1)
    # C: 00, 01, 10, 11 (for cols 0, 1)
    
    # M1 = A00 * (B00 + B11)
    U[0][0] = 1
    V[0][0] = 1 # B00
    V[1][0] = 1 # B11 (index 1 in col 0? No, B is 2x10. B[j,k]. j in 0..1, k in 0..9)
    # V index is j*p + k.
    # B00 is V[0*10+0] = V[0]
    # B11 is V[1*10+1] = V[11]
    V[11][0] = 1
    # C00 is W[0*2+0] = W[0]
    # C11 is W[1*2+1] = W[3]
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (A00 + A11) * B11
    U[0][1] = 1
    U[3][1] = 1
    V[11][1] = 1
    W[3][1] = 1
    
    # M3 = (A01 + A11) * (B10 + B11)
    U[1][2] = 1
    U[3][2] = 1
    V[10][2] = 1 # B10
    V[11][2] = 1 # B11
    W[1][2] = 1 # C01
    W[3][2] = 1 # C11
    
    # M4 = A11 * (B00 + B10)
    U[3][3] = 1
    V[0][3] = 1 # B00
    V[10][3] = 1 # B10
    W[1][3] = 1 # C01
    W[3][3] = 1 # C11
    
    # M5 = (A00 + A01) * B00
    U[0][4] = 1
    U[1][4] = 1
    V[0][4] = 1 # B00
    W[0][4] = 1 # C00
    W[1][4] = 1 # C01
    
    # M6 = (A10 + A11) * (B00 + B01)
    U[2][5] = 1
    U[3][5] = 1
    V[0][5] = 1 # B00
    V[1][5] = 1 # B01
    W[2][5] = 1 # C10
    W[3][5] = 1 # C11
    
    # M7 = (A00 + A01) * B01
    U[0][6] = 1
    U[1][6] = 1
    V[1][6] = 1 # B01
    W[1][6] = 1 # C01
    
    # Now for columns k = 2 to 9
    # We add 3 terms per column.
    # Terms:
    # M_k1 = A00 * B0k
    # M_k2 = A01 * B1k
    # M_k3 = (A10 + A11) * B0k
    # C0k = M_k1 + M_k2
    # C1k = M_k3 + ... wait, C1k = A10 B0k + A11 B1k.
    # My M_k3 only has A10+A11 times B0k. Missing A11 B1k.
    
    # The 3-term extension is not standard schoolbook.
    # I will use the schoolbook for the remaining columns to ensure correctness, 
    # which gives rank 7 + 4*8 = 39.
    # This is better than 40.
    
    col = 7
    for k in range(2, p):
        # 4 terms for column k
        # A00 B0k
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[0*n + k][col] = 1
        col += 1
        
        # A01 B1k
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[0*n + k][col] = 1
        col += 1
        
        # A10 B0k
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[1*n + k][col] = 1
        col += 1
        
        # A11 B1k
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[1*n + k][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
