"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose p=15 into seven 2s and one 1
    # Rank = 7 * 7 (Strassen for 2x2x2) + 4 (Schoolbook for 2x2x1) = 49 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)] # 4 x 53
    V = [[0] * rank for _ in range(m * p)] # 30 x 53
    W = [[0] * rank for _ in range(p * n)] # 30 x 53
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A is 2x2, B is 2x2 (cols k0, k1), C is 2x2
    # Strassen factors:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    
    # Indices in U: i*m + j
    # Indices in V: j*p + k
    # Indices in W: k*n + i
    
    def add_strassen_block(k0, k1, start_col):
        nonlocal col
        # 7 terms
        # Term 1: (A00+A11) * (B00+B11) -> C00 += M1, C01 += M1-M5, C10 += M4+M1, C11 += M1-M3
        # We need to map these to U, V, W entries.
        
        # U entries for A00+A11:
        # U[0, c] = 1, U[3, c] = 1
        # V entries for B00+B11:
        # V[0*p+k0, c] = 1 (B00), V[1*p+k1, c] = 1 (B11)
        # W entries for result:
        # This term contributes to C00, C01, C10, C11.
        # W[k*n+i, c].
        # C00 (i=0, k=k0): W[k0*2+0, c] += 1
        # C01 (i=0, k=k1): W[k1*2+0, c] += -1? No, the signs are in the linear combination.
        
        # Let's just write the 7 rank-1 tensors explicitly.
        
        # 1. M1
        U[0][start_col] += 1
        U[3][start_col] += 1
        V[k0][start_col] += 1 # B00 is at row 0*p+k0 = k0
        V[p+k1][start_col] += 1 # B11 is at row 1*p+k1 = p+k1
        W[k0][start_col] += 1 # C00
        W[k1][start_col] -= 1 # C01 (M1-M5, so M1 part is +1)
        W[2+k0][start_col] += 1 # C10
        W[2+k1][start_col] -= 1 # C11 (M1-M3, so M1 part is +1)
        col += 1

        # 2. M2
        U[2][start_col] += 1 # A10
        U[3][start_col] += 1 # A11
        V[k0][start_col] += 1 # B00
        W[2+k0][start_col] += 1 # C10
        col += 1

        # 3. M3
        U[0][start_col] += 1 # A00
        V[p+k0][start_col] += 1 # B01 (row 1*p+k0)
        V[p+k1][start_col] -= 1 # B11 (row 1*p+k1)
        W[2+k1][start_col] += 1 # C11 (M1-M3, so -M3)
        col += 1

        # 4. M4
        U[3][start_col] += 1 # A11
        V[k1][start_col] += 1 # B10 (row 0*p+k1)
        V[k0][start_col] -= 1 # B00 (row 0*p+k0)
        W[2+k0][start_col] += 1 # C10
        col += 1

        # 5. M5
        U[0][start_col] += 1 # A00
        U[1][start_col] += 1 # A01
        V[p+k1][start_col] += 1 # B11
        W[k1][start_col] += 1 # C01
        col += 1

        # 6. M6
        U[2][start_col] += 1 # A10
        U[0][start_col] -= 1 # A00
        V[k0][start_col] += 1 # B00
        V[k1][start_col] += 1 # B10
        W[k0][start_col] += 1 # C00
        col += 1

        # 7. M7
        U[1][start_col] += 1 # A01
        U[3][start_col] -= 1 # A11
        V[p+k0][start_col] += 1 # B01
        V[p+k1][start_col] += 1 # B11
        W[k1][start_col] += 1 # C01
        col += 1

    # 7 blocks of 2x2x2
    for block_idx in range(7):
        k0 = block_idx * 2
        k1 = k0 + 1
        add_strassen_block(k0, k1, col)
        
    # 1 block of 2x2x1 (col 14)
    k0 = 14
    # Schoolbook: A00*B00, A01*B10, A10*B00, A11*B10
    # C00 = A00*B00
    U[0][col] += 1
    V[k0][col] += 1
    W[k0][col] += 1
    col += 1
    
    # C01 = A01*B10 (Result col 14, row 0? No, result is 2x15. Col 14 is the only col in this block)
    # Wait, the block is 2x2x1. B is 2x1.
    # B00 is B[0, 14]. B10 is B[1, 14].
    # C[0, 14] = A[0,0]B[0,14] + A[0,1]B[1,14]
    # C[1, 14] = A[1,0]B[0,14] + A[1,1]B[1,14]
    
    # Term 2: A01 * B10
    U[1][col] += 1
    V[p+k0][col] += 1 # B10 is at row 1*p+14
    W[k0][col] += 1
    col += 1
    
    # Term 3: A10 * B00
    U[2][col] += 1
    V[k0][col] += 1
    W[2+k0][col] += 1
    col += 1
    
    # Term 4: A11 * B10
    U[3][col] += 1
    V[p+k0][col] += 1
    W[2+k0][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
