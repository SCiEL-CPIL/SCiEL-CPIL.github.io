"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)
    
    # Strassen 2x2: 7 multiplications
    # Let's decompose into blocks of size 2x2 for A, 2x4 for B, 2x4 for C
    # A is 4x4, B is 4x8, C is 4x8
    # Block partition:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x4
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x4
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # For each 2x2 * 2x4 multiplication, we can use 7 multiplications.
    # A 2x2 * 2x4 is like multiplying a 2x2 matrix by a 2x4 matrix.
    # We can view this as two 2x2 * 2x2 multiplications (for the first two cols and last two cols).
    # So for each 2x2 A block and 2x4 B block, we need 14 multiplications (7 for each 2x2 subblock).
    #
    # Total blocks: 4 A blocks, each multiplies with 2 B blocks (top and bottom rows of B)
    # Wait, let me re-think.
    # A00 (2x2) * B00 (2x4) -> 14 mults
    # A00 (2x2) * B01 (2x4) -> 14 mults
    # A01 (2x2) * B10 (2x4) -> 14 mults
    # A01 (2x2) * B11 (2x4) -> 14 mults
    # A10 (2x2) * B00 (2x4) -> 14 mults
    # A10 (2x2) * B01 (2x4) -> 14 mults
    # A11 (2x2) * B10 (2x4) -> 14 mults
    # A11 (2x2) * B11 (2x4) -> 14 mults
    # Total: 8 * 14 = 112
    # That's only slightly better than 128. Not great.
    
    # Let's try a different approach. Use Strassen for the whole 4x4*4x8.
    # Actually, let's just do a simple block decomposition with 2x2 blocks.
    # For each 2x2 A block and each 2x8 B block (rows), do Strassen-like.
    
    # Actually, let me just implement the standard Strassen for 2x2 blocks
    # and tile them.
    
    # U indices: i*m + j, where i is row of A (0..3), j is col of A (0..3)
    # V indices: j*p + k, where j is row of B (0..3), k is col of B (0..7)
    # W indices: k*n + i, where k is col of C (0..7), i is row of C (0..3)
    
    U = [[0] * 0 for _ in range(n * m)]
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    col = 0
    # For each 2x2 block of A (there are 4 such blocks: (0,0), (0,1), (1,0), (1,1))
    # and for each pair of rows in B (2x8), do 7 Strassen multiplications.
    # Actually, let's do it more carefully.
    
    # Let's use 2x2 blocks for A, and for B, we have 2x8 blocks.
    # A block (bi, bj) where bi, bj in {0, 1}
    # B block (bi, *) where bi in {0, 1}, covering rows 2*bi..2*bi+1 of B, all 8 cols
    # C block (bi, *) where bi in {0, 1}, covering rows 2*bi..2*bi+1 of C, all 8 cols
    
    for ai in range(2):  # A row block
        for aj in range(2):  # A col block
            for bi in range(2):  # B row block
                # A subblock: rows 2*ai..2*ai+1, cols 2*aj..2*aj+1
                # B subblock: rows 2*bi..2*bi+1, cols 0..7
                # C subblock: rows 2*ai..2*ai+1, cols 0..7
                #
                # We need to compute C_sub = A_sub * B_sub
                # A_sub is 2x2, B_sub is 2x8, C_sub is 2x8
                #
                # We can split B_sub into two 2x4 blocks: B_left (cols 0..3) and B_right (cols 4..7)
                # And C_sub into C_left (cols 0..3) and C_right (cols 4..7)
                #
                # C_left = A_sub * B_left  (2x2 * 2x4)
                # C_right = A_sub * B_right  (2x2 * 2x4)
                
                for k_block in range(2):  # left (0) or right (1)
                    k_offset = 4 * k_block
                    # B_sub: rows 2*bi..2*bi+1, cols k_offset..k_offset+3
                    # This is a 2x4 matrix, which we view as two 2x2 matrices:
                    # B_col0: cols k_offset, k_offset+1
                    # B_col1: cols k_offset+2, k_offset+3
                    
                    # Strassen for 2x2 * 2x4 (two 2x2 multiplications)
                    # Let A = [[a, b], [c, d]] (2x2)
                    # B = [[e, f, g, h], [i, j, k, l]] (2x4)
                    # We want C = A*B (2x4)
                    # 
                    # Using Strassen on each 2x2 subblock:
                    # For B_cols0 = [[e, f], [i, j]]:
                    # M1 = (a+d)(e+j)
                    # M2 = (c+d)(e)
                    # M3 = a(f-j)
                    # M4 = d(i-e)
                    # M5 = (a+b)(j)
                    # M6 = (c-a)(e+f)
                    # M7 = (b-d)(i+j)
                    #
                    # C00 = M1 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M4
                    #
                    # Similarly for B_cols1 = [[g, h], [k, l]]:
                    # M8 = (a+d)(g+l)
                    # M9 = (c+d)(g)
                    # M10 = a(h-l)
                    # M11 = d(k-g)
                    # M12 = (a+b)(l)
                    # M13 = (c-a)(g+h)
                    # M14 = (b-d)(k+l)
                    #
                    # C02 = M8 - M12 + M14
                    # C03 = M10 + M12
                    # C12 = M9 + M11
                    # C13 = M8 - M9 + M10 + M11
                    
                    # That's 14 multiplications per (ai, aj, bi, k_block)
                    # Total: 2*2*2*2 * 14 = 112
                    
                    # Define the 14 Strassen terms
                    # U indices: A rows 2*ai, 2*ai+1; A cols 2*aj, 2*aj+1
                    a00 = (2*ai) * m + (2*aj)      # a
                    a01 = (2*ai) * m + (2*aj+1)    # b
                    a10 = (2*ai+1) * m + (2*aj)    # c
                    a11 = (2*ai+1) * m + (2*aj+1)  # d
                    
                    # B rows 2*bi, 2*bi+1
                    b00 = (2*bi) * p + (k_offset)      # e
                    b01 = (2*bi) * p + (k_offset+1)    # f
                    b02 = (2*bi) * p + (k_offset+2)    # g
                    b03 = (2*bi) * p + (k_offset+3)    # h
                    b10 = (2*bi+1) * p + (k_offset)    # i
                    b11 = (2*bi+1) * p + (k_offset+1)  # j
                    b12 = (2*bi+1) * p + (k_offset+2)  # k
                    b13 = (2*bi+1) * p + (k_offset+3)  # l
                    
                    # C rows 2*ai, 2*ai+1
                    c00 = k_offset * n + (2*ai)
                    c01 = (k_offset+1) * n + (2*ai)
                    c02 = (k_offset+2) * n + (2*ai)
                    c03 = (k_offset+3) * n + (2*ai)
                    c10 = k_offset * n + (2*ai+1)
                    c11 = (k_offset+1) * n + (2*ai+1)
                    c12 = (k_offset+2) * n + (2*ai+1)
                    c13 = (k_offset+3) * n + (2*ai+1)
                    
                    # M1 = (a+d)(e+j)
                    terms = [
                        # (U_indices, V_indices, W_indices, coefficient)
                        # M1
                        ([a00, a11], [b00, b11], [c00, c13, c11], [1, -1, 1]),  # C00+=M1, C13+=M1, C11+=M1? No, let me re-derive
                    ]
                    # Actually, let me just directly implement the 14 multiplications
                    # and accumulate into U, V, W.
                    
                    # This is getting complex. Let me just use the schoolbook but with a better structure.
                    pass
    
    # Given the complexity, let me just return the schoolbook for now.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
