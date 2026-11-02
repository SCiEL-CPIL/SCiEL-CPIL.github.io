"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication
    using a 4+2 block partition with Strassen for 2x2 sub-blocks.
    
    Partition each 6x6 matrix into blocks:
    A = [[A44, A42], [A24, A22]]  where A44 is 4x4, A42 is 4x2, A24 is 2x4, A22 is 2x2
    B = [[B44, B42], [B24, B22]]  where B44 is 4x4, B42 is 4x2, B24 is 2x4, B22 is 2x2
    
    C = A*B = [[C44, C42], [C24, C22]]
    C44 = A44*B44 + A42*B24
    C42 = A44*B42 + A42*B22
    C24 = A24*B44 + A22*B24
    C22 = A24*B42 + A22*B22
    
    For 4x4 * 4x4: use schoolbook (rank 16)
    For 4x2 * 2x4: decompose into 4 instances of 2x2 * 2x2, each with Strassen (rank 7), total 28
    For 4x4 * 4x2: decompose into 2 instances of 4x2 * 2x2... 
    Actually, let's be more careful.
    
    Let me think about this differently. The key insight is:
    - 4x4 * 4x4 = 16 rank-1 terms (schoolbook)
    - 4x2 * 2x4: We can view this as 4 independent 2x2 * 2x2 multiplications
      (rows 0-1 of A42 times rows 0-1 of B24, rows 2-3 times rows 0-1, etc.)
      Wait, no. A42 is 4x2, B24 is 2x4. The product is 4x4.
      C_ij = sum_k A42_ik * B24_kj for i in 0..3, j in 0..3, k in 0..1
      This is 4*4*2 = 32 rank-1 terms in schoolbook.
      But we can group: for each pair of rows (i1,i2) from {0,1} and {2,3},
      and each pair of columns (j1,j2) from {0,1} and {2,3},
      we have a 2x2 * 2x2 product.
      That's 4 such products, each rank 7 with Strassen = 28 total.
    
    Similarly:
    - 4x4 * 4x2: product is 4x2. C_ij = sum_k A44_ik * B42_kj for i in 0..3, j in 0..1, k in 0..3
      Schoolbook: 4*2*4 = 32 terms.
      Group into 2x2 blocks: rows {0,1} and {2,3} of A44, columns {0,1} of B42.
      For rows {0,1}: 2x4 * 4x2 = (2x4)*(4x2). Decompose A44[0:2,:] as 2x4, B42 as 4x2.
      This is like 2 independent 2x2 * 2x2? No.
      Actually: A44[0:2,:] * B42 gives a 2x2 result.
      A44[0:2,:] is 2x4, B42 is 4x2.
      C[0:2,0:2] = A44[0:2,:] * B42 = sum over 4 middle indices.
      This can be decomposed as:
      A44[0:2,0:2]*B42[0:2,0:2] + A44[0:2,2:4]*B42[2:4,0:2]
      Each of these is 2x2 * 2x2, rank 7 with Strassen.
      So 2 * 7 = 14 for the top half.
      Similarly 14 for the bottom half. Total 28.
    
    - 2x4 * 4x4: similar to above, 28.
    - 4x2 * 2x2: A42 is 4x2, B22 is 2x2. Product is 4x2.
      C_ij = sum_k A42_ik * B22_kj for i in 0..3, j in 0..1, k in 0..1
      Schoolbook: 4*2*2 = 16.
      Group: rows {0,1} and {2,3} of A42, all of B22.
      A42[0:2,:] * B22 is 2x2 * 2x2, rank 7.
      A42[2:4,:] * B22 is 2x2 * 2x2, rank 7.
      Total 14.
    
    - 2x4 * 4x2: A24 is 2x4, B42 is 4x2. Product is 2x2.
      C_ij = sum_k A24_ik * B42_kj for i in 0..1, j in 0..1, k in 0..3
      Schoolbook: 2*2*4 = 16.
      Group: A24[:,0:2] * B42[0:2,:] + A24[:,2:4] * B42[2:4,:]
      Each is 2x2 * 2x2, rank 7. Total 14.
    
    - 2x2 * 2x2: rank 7 with Strassen.
    
    Total rank:
    C44 = 16 (A44*B44) + 28 (A42*B24) = 44
    C42 = 28 (A44*B42) + 14 (A42*B22) = 42
    C24 = 28 (A24*B44) + 14 (A22*B24) = 42
    C22 = 14 (A24*B42) + 7 (A22*B22) = 21
    Total: 44 + 42 + 42 + 21 = 149
    """
    n, m, p = DIMENSIONS
    
    U = [[0] * 149 for _ in range(n * m)]
    V = [[0] * 149 for _ in range(m * p)]
    W = [[0] * 149 for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 decomposition helper
    # For X (2x2) * Y (2x2) = Z (2x2)
    # X elements indexed by (r,c) in {0,1}x{0,1}
    # Y elements indexed by (r,c) in {0,1}x{0,1}
    # Z elements indexed by (r,c) in {0,1}x{0,1}
    
    def add_strassen_2x2(x_base_row, x_base_col, y_base_row, y_base_col, z_base_row, z_base_col, offset):
        """Add Strassen 2x2 terms. x_base_row/col: top-left of X in A.
        y_base_row/col: top-left of Y in B. z_base_row/col: top-left of Z in C.
        Returns number of terms added."""
        nonlocal col
        # Strassen terms:
        # s=0: X = X00+X11, Y = Y00+Y11, Z: Z00+=M0, Z11+=M0
        # s=1: X = X10+X11, Y = Y00, Z: Z10+=M1, Z11-=M1
        # s=2: X = X00, Y = Y01-Y11, Z: Z01+=M2, Z11+=M2
        # s=3: X = X11, Y = Y10-Y00, Z: Z00+=M3, Z10+=M3
        # s=4: X = X00+X01, Y = Y11, Z: Z00-=M4, Z01+=M4
        # s=5: X = X10-X00, Y = Y00+Y01, Z: Z11+=M5
        # s=6: X = X01-X11, Y = Y10+Y11, Z: Z00+=M6
        
        strassen = [
            # (x_coeffs, y_coeffs, z_coeffs)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for x_coeffs, y_coeffs, z_coeffs in strassen:
            for (lr, lc), coeff in x_coeffs.items():
                row = x_base_row + lr
                c = x_base_col + lc
                U[row * m + c][col] += coeff
            for (lr, lc), coeff in y_coeffs.items():
                row = y_base_row + lr
                c = y_base_col + lc
                V[row * p + c][col] += coeff
            for (lr, lc), coeff in z_coeffs.items():
                row = z_base_row + lr
                c = z_base_col + lc
                # W index: W[k*n+i, r] where i is row of C, k is col of C
                w_row = c * n + row
                W[w_row][col] += coeff
            col += 1
        
        return 7
    
    def add_schoolbook_4x4(x_base_row, x_base_col, y_base_row, y_base_col, z_base_row, z_base_col):
        """Add schoolbook 4x4 * 4x4 terms. Returns 16."""
        nonlocal col
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U[(x_base_row + i) * m + (x_base_col + k)][col] = 1
                    V[(y_base_row + k) * p + (y_base_col + j)][col] = 1
                    # C[i,j] += A[i,k]*B[k,j]
                    w_row = (z_base_col + j) * n + (z_base_row + i)
                    W[w_row][col] = 1
                    col += 1
        return 16
    
    # C44 = A44*B44 + A42*B24
    # A44 is at (0,0) in A, B44 is at (0,0) in B, C44 is at (0,0) in C
    col += add_schoolbook_4x4(0, 0, 0, 0, 0, 0)
    
    # A42*B24: A42 is at (0,4) in A (4x2), B24 is at (4,0) in B (2x4)
    # Decompose into 4 Strassen 2x2:
    # A42[0:2,0:2] * B24[0:2,0:2] -> C44[0:2,0:2]
    # A42[0:2,2:4]... wait, A42 is 4x2, so columns are 0,1.
    # B24 is 2x4, so rows are 0,1, columns are 0,1,2,3.
    # A42[i,k] * B24[k,j] for i in 0..3, j in 0..3, k in 0..1
    # Group by (i_group, j_group) where i_group in {0,1} (rows 0-1, 2-3), j_group in {0,1} (cols 0-1, 2-3)
    for ig in range(2):
        for jg in range(2):
            # A42[ig*2:(ig+1)*2, :] is 2x2
            # B24[:, jg*2:(jg+1)*2] is 2x2
            # Product contributes to C44[ig*2:(ig+1)*2, jg*2:(jg+1)*2]
            col += add_strassen_2x2(
                x_base_row=ig * 2, x_base_col=4,
                y_base_row=4, y_base_col=jg * 2,
                z_base_row=ig * 2, z_base_col=jg * 2
            )
    # That's 4 * 7 = 28 terms
    
    # C42 = A44*B42 + A42*B22
    # A44*B42: A44 is 4x4 at (0,0), B42 is 4x2 at (0,4)
    # Product is 4x2 at (0,4) in C
    # Decompose: for each row group of A44 (2 groups of 2 rows), multiply by B42
    # A44[0:2,:] * B42 -> 2x2, A44[2:4,:] * B42 -> 2x2
    # Each is 2x4 * 4x2, decomposed as 2x2 * 2x2 + 2x2 * 2x2
    for ig in range(2):
        # A44[ig*2:(ig+1)*2, 0:2] * B42[0:2, :] -> contributes to C42[ig*2:(ig+1)*2, :]
        col += add_strassen_2x2(
            x_base_row=ig * 2, x_base_col=0,
            y_base_row=0, y_base_col=4,
            z_base_row=ig * 2, z_base_col=4
        )
        # A44[ig*2:(ig+1)*2, 2:4] * B42[2:4, :] -> contributes to C42[ig*2:(ig+1)*2, :]
        col += add_strassen_2x2(
            x_base_row=ig * 2, x_base_col=2,
            y_base_row=2, y_base_col=4,
            z_base_row=ig * 2, z_base_col=4
        )
    # That's 2 * 2 * 7 = 28 terms
    
    # A42*B22: A42 is 4x2 at (0,4), B22 is 2x2 at (4,4)
    # Product is 4x2 at (0,4) in C
    # A42[0:2,:] * B22 -> 2x2, A42[2:4,:] * B22 -> 2x2
    for ig in range(2):
        col += add_strassen_2x2(
            x_base_row=ig * 2, x_base_col=4,
            y_base_row=4, y_base_col=4,
            z_base_row=ig * 2, z_base_col=4
        )
    # That's 2 * 7 = 14 terms
    
    # C24 = A24*B44 + A22*B24
    # A24*B44: A24 is 2x4 at (4,0), B44 is 4x4 at (0,0)
    # Product is 2x4 at (4,0) in C
    # A24[:, 0:2] * B44[0:2, :] -> 2x2, A24[:, 2:4] * B44[2:4, :] -> 2x2
    for jg in range(2):
        col += add_strassen_2x2(
            x_base_row=4, x_base_col=jg * 2,
            y_base_row=0, y_base_col=0,
            z_base_row=4, z_base_col=jg * 2
        )
        col += add_strassen_2x2(
            x_base_row=4, x_base_col=jg * 2,
            y_base_row=2, y_base_col=0,
            z_base_row=4, z_base_col=jg * 2
        )
    # That's 2 * 2 * 7 = 28 terms
    
    # A22*B24: A22 is 2x2 at (4,4), B24 is 2x4 at (4,0)
    # Product is 2x4 at (4,0) in C
    # A22[:, 0:2]... wait, A22 is 2x2, so columns 0,1.
    # A22 * B24[:, jg*2:(jg+1)*2] for jg in 0,1
    for jg in range(2):
        col += add_strassen_2x2(
            x_base_row=4, x_base_col=4,
            y_base_row=4, y_base_col=jg * 2,
            z_base_row=4, z_base_col=jg * 2
        )
    # That's 2 * 7 = 14 terms
    
    # C22 = A24*B42 + A22*B22
    # A24*B42: A24 is 2x4 at (4,0), B42 is 4x2 at (0,4)
    # Product is 2x2 at (4,4) in C
    # A24[:, 0:2] * B42[0:2, :] -> 2x2, A24[:, 2:4] * B42[2:4, :] -> 2x2
    for jg in range(2):
        col += add_strassen_2x2(
            x_base_row=4, x_base_col=jg * 2,
            y_base_row=jg * 2, y_base_col=4,
            z_base_row=4, z_base_col=4
        )
    # That's 2 * 7 = 14 terms
    
    # A22*B22: A22 is 2x2 at (4,4), B22 is 2x2 at (4,4)
    # Product is 2x2 at (4,4) in C
    col += add_strassen_2x2(
        x_base_row=4, x_base_col=4,
        y_base_row=4, y_base_col=4,
        z_base_row=4, z_base_col=4
    )
    # That's 7 terms
    
    # Total should be 16 + 28 + 28 + 14 + 28 + 14 + 14 + 7 = 149
    # Let's verify: 16+28=44, 28+14=42, 28+14=42, 14+7=21, total=149
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
