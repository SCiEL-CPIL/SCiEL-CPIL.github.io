"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the p=10 dimension into 5 blocks of size 2.
    # For each block, we multiply A (2x3) by B_block (3x2).
    # We decompose this 2x3x2 multiplication into:
    # 1. A 2x2x2 multiplication (using first 2 cols of A, first 2 rows of B_block) -> Rank 7 (Strassen)
    # 2. A 2x1x2 multiplication (using last col of A, last row of B_block) -> Rank 4 (Schoolbook)
    # Total rank per block = 11.
    # Total rank = 5 * 11 = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Indices in U for A (2x3):
    # a00: 0*3+0 = 0
    # a01: 0*3+1 = 1
    # a02: 0*3+2 = 2
    # a10: 1*3+0 = 3
    # a11: 1*3+1 = 4
    # a12: 1*3+2 = 5
    
    for block in range(5):
        # B_block columns are 2*block and 2*block+1
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # --- Strassen Part (2x2x2) ---
        # Uses A cols 0,1 and B rows 0,1
        # B indices:
        # b00: 0*10 + k0
        # b01: 0*10 + k1
        # b10: 1*10 + k0
        # b11: 1*10 + k1
        
        # M1
        U[0][col] = 1; U[4][col] = 1
        V[0*10+k0][col] = 1; V[0*10+k1][col] = 1
        W[k0*2+0][col] = 1; W[k1*2+0][col] = 1
        W[k0*2+1][col] = 1; W[k1*2+1][col] = 1
        # Wait, W indices are k*n + i.
        # C00 gets M1. C11 gets M1.
        # C00 index: k0*2 + 0. C11 index: k1*2 + 1.
        # M1 contributes to C00 and C11.
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So W for M1:
        # W[k0*2+0] += 1
        # W[k1*2+1] += 1
        # But I am writing to specific column 'col'.
        # I need to set W[k0*2+0][col] = 1 and W[k1*2+1][col] = 1.
        # And for other W indices, 0.
        
        # Let's rewrite carefully.
        # U, V, W are matrices.
        # For each multiplication c, we set U[:, c], V[:, c], W[:, c].
        
        # M1
        U[0][col] = 1; U[4][col] = 1
        V[0*10+k0][col] = 1; V[0*10+k1][col] = 1
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M2
        U[3][col] = 1; U[4][col] = 1
        V[0*10+k0][col] = 1
        W[k0*2+1][col] = -1
        W[k1*2+1][col] = 1 # C11 = ... - M2 ...
        # Wait, C11 = M1 - M2 + M3 + M6.
        # So W[k1*2+1] for M2 is -1.
        # C10 = M2 + M4. So W[k0*2+1] for M2 is 1.
        W[k0*2+1][col] = 1
        W[k1*2+1][col] = -1
        col += 1
        
        # M3
        U[0][col] = 1
        V[0*10+k1][col] = 1; V[1*10+k1][col] = -1
        W[k1*2+0][col] = 1 # C01 = M3 + M5
        W[k1*2+1][col] = 1 # C11 = ... + M3 ...
        col += 1
        
        # M4
        U[4][col] = 1
        V[1*10+k0][col] = 1; V[0*10+k0][col] = -1
        W[k0*2+0][col] = 1 # C00 = ... + M4 ...
        W[k0*2+1][col] = 1 # C10 = ... + M4 ...
        col += 1
        
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[1*10+k1][col] = 1
        W[k0*2+0][col] = -1 # C00 = ... - M5 ...
        W[k1*2+0][col] = 1 # C01 = ... + M5 ...
        col += 1
        
        # M6
        U[3][col] = 1; U[0][col] = -1
        V[0*10+k0][col] = 1; V[0*10+k1][col] = 1
        W[k1*2+1][col] = 1 # C11 = ... + M6 ...
        col += 1
        
        # M7
        U[1][col] = 1; U[4][col] = -1
        V[1*10+k0][col] = 1; V[1*10+k1][col] = 1
        W[k0*2+0][col] = 1 # C00 = ... + M7 ...
        col += 1

        # --- Schoolbook Part (2x1x2) ---
        # Uses A col 2 and B row 2
        # A indices: 2 (a02), 5 (a12)
        # B indices: 2*10+k0 (b20), 2*10+k1 (b21)
        
        # M8 = a02 * b20
        U[2][col] = 1
        V[2*10+k0][col] = 1
        W[k0*2+0][col] = 1
        col += 1
        
        # M9 = a02 * b21
        U[2][col] = 1
        V[2*10+k1][col] = 1
        W[k1*2+0][col] = 1
        col += 1
        
        # M10 = a12 * b20
        U[5][col] = 1
        V[2*10+k0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # M11 = a12 * b21
        U[5][col] = 1
        V[2*10+k1][col] = 1
        W[k1*2+1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
