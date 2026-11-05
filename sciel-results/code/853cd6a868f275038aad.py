"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll decompose into:
    # Part 1: 4x4 x 4x4 (left 4 cols of B) using rank-49 Strassen
    # Part 2: 4x4 x 4x3 (right 3 cols of B) using schoolbook rank 48
    # Total rank = 49 + 48 = 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # Strassen 2x2 coefficients
    # For product Z = X*Y where X,Y are 2x2:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X11-X01)*(Y10+Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Block Strassen for 4x4:
    # S1 = (A00+A11)*(B00+B11)
    # S2 = (A10+A11)*B00
    # S3 = A00*(B01-B11)
    # S4 = A11*(B10-B00)
    # S5 = (A00+A01)*B11
    # S6 = (A10-A00)*(B00+B01)
    # S7 = (A11-A01)*(B10+B11)
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Each Ss is a 2x2 product computed with 7 scalar Strassen mults.
    # So we have 7*7 = 49 scalar mults total for the 4x4 part.
    
    # Let's index the scalar mults as (s, t) where s in 0..6 is the block Strassen
    # term and t in 0..6 is the inner Strassen term.
    
    # Block Strassen coefficients for A-blocks (row, col) -> coefficient in term s
    # A is indexed as A[row_block][col_block] where row_block, col_block in {0,1}
    # Each block is 2x2, so A[row_block][col_block] has entries A[2*rb+i][2*cb+j]
    
    # For term s, the A-block combination:
    # s=0: A00+A11 -> blocks (0,0) and (1,1)
    # s=1: A10+A11 -> blocks (1,0) and (1,1)
    # s=2: A00 -> block (0,0)
    # s=3: A11 -> block (1,1)
    # s=4: A00+A01 -> blocks (0,0) and (0,1)
    # s=5: A10-A00 -> blocks (1,0) and (0,0) with -1
    # s=6: A11-A01 -> blocks (1,1) and (0,1) with -1
    
    # For term s, the B-block combination:
    # s=0: B00+B11 -> blocks (0,0) and (1,1)
    # s=1: B00 -> block (0,0)
    # s=2: B01-B11 -> blocks (0,1) and (1,1) with -1
    # s=3: B10-B00 -> blocks (1,0) and (0,0) with -1
    # s=4: B11 -> block (1,1)
    # s=5: B00+B01 -> blocks (0,0) and (0,1)
    # s=6: B10+B11 -> blocks (1,0) and (1,1)
    
    # Define which blocks are used in each block Strassen term for A:
    # a_block_coeffs[s] = list of (row_block, col_block, coeff)
    a_block_coeffs = [
        [(0,0,1), (1,1,1)],      # s=0
        [(1,0,1), (1,1,1)],      # s=1
        [(0,0,1)],                # s=2
        [(1,1,1)],                # s=3
        [(0,0,1), (0,1,1)],      # s=4
        [(1,0,1), (0,0,-1)],     # s=5
        [(1,1,1), (0,1,-1)],     # s=6
    ]
    
    # b_block_coeffs[s] = list of (row_block, col_block, coeff)
    b_block_coeffs = [
        [(0,0,1), (1,1,1)],      # s=0
        [(0,0,1)],                # s=1
        [(0,1,1), (1,1,-1)],     # s=2
        [(1,0,1), (0,0,-1)],     # s=3
        [(1,1,1)],                # s=4
        [(0,0,1), (0,1,1)],      # s=5
        [(1,0,1), (1,1,1)],      # s=6
    ]
    
    # Inner Strassen for 2x2: for X*Y -> Z
    # Inner term t contributes to X blocks (x_rb, x_cb) and Y blocks (y_rb, y_cb)
    # and Z entries (z_rb, z_cb)
    
    # For inner Strassen term t, which 2x2 entries of X and Y are used:
    # t=0: X00+X11, Y00+Y11 -> Z00, Z11
    # t=1: X10+X11, Y00 -> Z10, Z11
    # t=2: X00, Y01-Y11 -> Z01, Z11
    # t=3: X11, Y10-Y00 -> Z00, Z10
    # t=4: X00+X01, Y11 -> Z00, Z01
    # t=5: X10-X00, Y00+Y01 -> Z11
    # t=6: X11-X01, Y10+Y11 -> Z00, Z11
    
    # For the inner Strassen, X is a 2x2 matrix with entries X[0,0], X[0,1], X[1,0], X[1,1]
    # These correspond to positions in the block. If the block is at position (rb, cb) in the 4x4,
    # then X[i,j] = A[2*rb+i, 2*cb+j]
    
    # Inner Strassen term t: X entries used (with coeffs), Y entries used (with coeffs)
    x_inner_coeffs = [
        [(0,0,1), (1,1,1)],      # t=0
        [(1,0,1), (1,1,1)],      # t=1
        [(0,0,1)],                # t=2
        [(1,1,1)],                # t=3
        [(0,0,1), (0,1,1)],      # t=4
        [(1,0,1), (0,0,-1)],     # t=5
        [(1,1,1), (0,1,-1)],     # t=6
    ]
    
    y_inner_coeffs = [
        [(0,0,1), (1,1,1)],      # t=0
        [(0,0,1)],                # t=1
        [(0,1,1), (1,1,-1)],     # t=2
        [(1,0,1), (0,0,-1)],     # t=3
        [(1,1,1)],                # t=4
        [(0,0,1), (0,1,1)],      # t=5
        [(1,0,1), (1,1,1)],      # t=6
    ]
    
    # Z entries contributed by inner term t:
    # t=0: Z00(+1), Z11(+1)
    # t=1: Z10(+1), Z11(-1)
    # t=2: Z01(+1), Z11(+1)
    # t=3: Z00(+1), Z10(+1)
    # t=4: Z00(-1), Z01(+1)
    # t=5: Z11(+1)
    # t=6: Z00(+1), Z11(+1)
    
    z_inner_coeffs = [
        [(0,0,1), (1,1,1)],      # t=0
        [(1,0,1), (1,1,-1)],     # t=1
        [(0,1,1), (1,1,1)],      # t=2
        [(0,0,1), (1,0,1)],      # t=3
        [(0,0,-1), (0,1,1)],     # t=4
        [(1,1,1)],                # t=5
        [(0,0,1), (1,1,1)],      # t=6
    ]
    
    # Block Strassen: which Z blocks (row_block, col_block) get which S terms
    # C00 = S1 + S4 - S5 + S7 -> S0(+1), S3(+1), S4(-1), S6(+1)
    # C01 = S3 + S5 -> S2(+1), S4(+1)
    # C10 = S2 + S4 -> S1(+1), S3(+1)
    # C11 = S1 - S2 + S3 + S6 -> S0(+1), S1(-1), S2(+1), S5(+1)
    
    z_block_coeffs = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],
    }
    
    # Now construct U, V, W for the 4x4x4 part (rank 49)
    col = 0
    for s in range(7):
        for t in range(7):
            # U entries: A[row, col] where row = 2*ar + ai, col = 2*ac + aj
            # For block Strassen term s, A blocks used with coeffs from a_block_coeffs[s]
            # For inner Strassen term t, within each block, entries used with coeffs from x_inner_coeffs[t]
            
            for (ar, ac, a_coeff) in a_block_coeffs[s]:
                for (ai, aj, xi_coeff) in x_inner_coeffs[t]:
                    row = 2*ar + ai
                    col_a = 2*ac + aj
                    U[row * m + col_a][col] = a_coeff * xi_coeff
            
            # V entries: B[row, col] where row = 2*br + bi, col = 2*bc + bj
            # But B is 4x7, so col goes 0..6. For the left 4 columns (cols 0-3):
            # B[row, col] where col = 2*bc + bj for bc in {0,1}, bj in {0,1}
            
            for (br, bc, b_coeff) in b_block_coeffs[s]:
                for (bi, bj, yi_coeff) in y_inner_coeffs[t]:
                    row_b = 2*br + bi
                    col_b = 2*bc + bj
                    V[row_b * p + col_b][col] = b_coeff * yi_coeff
            
            # W entries: C[row, col] where row = 2*cr + ci, col = 2*cc + cj
            # W is indexed as W[k*n + i] where k is col of C, i is row of C
            # C[row_c, col_c] where row_c = 2*cr + ci, col_c = 2*cc + cj
            # In W: index = col_c * n + row_c = (2*cc + cj)*4 + (2*cr + ci)
            
            for (cr, cc, z_block_coeff) in [(zb[0], zb[1], zb[2]) for zb in z_block_coeffs.keys()]:
                # Check if this (cr, cc) block uses term s with what coeff
                coeff_in_z = 0
                for (s_idx, zcoeff) in z_block_coeffs[(cr, cc)]:
                    if s_idx == s:
                        coeff_in_z = zcoeff
                        break
                if coeff_in_z == 0:
                    continue
                for (ci, cj, zi_coeff) in z_inner_coeffs[t]:
                    row_c = 2*cr + ci
                    col_c = 2*cc + cj
                    w_idx = col_c * n + row_c
                    W[w_idx][col] = coeff_in_z * zi_coeff
            
            col += 1
    
    # Part 2: Schoolbook for 4x4 x 4x3 (right 3 cols of B)
    # C[:, 4:7] = A * B[:, 4:7]
    # Rank = 4 * 4 * 3 = 48
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k = 0,1,2 for cols 4,5,6 of B
                row_u = i * m + j
                row_v = j * p + (4 + k)
                col_w = (4 + k) * n + i
                U[row_u][col] = 1
                V[row_v][col] = 1
                W[col_w][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
