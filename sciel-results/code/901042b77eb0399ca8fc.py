"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 65 decomposition.
    # 49 terms for 4x4x4 (Strassen)
    # 16 terms for 4x4x1 (Schoolbook)
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen 4x4x4 ---
    # Outer steps: 7
    # Each outer step has 7 inner steps.
    
    # Define outer steps (X, Y)
    # X, Y are 2x2 matrices of coefficients for A, B blocks.
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # We represent X as a 2x2 list of dicts {block_idx: coeff}
    # Or simply, we can trace the scalar coefficients.
    
    # Let's define the 7 outer steps explicitly.
    # Step 1: X = A00 + A11, Y = B00 + B11
    # Step 2: X = A10 + A11, Y = B00
    # Step 3: X = A00, Y = B01 - B11
    # Step 4: X = A11, Y = B10 - B00
    # Step 5: X = A00 + A01, Y = B11
    # Step 6: X = A10 - A00, Y = B00 + B01
    # Step 7: X = A01 - A11, Y = B10 + B11
    
    # For each outer step, we need the coefficients of A_ij and B_kl in X_ab and Y_ab.
    # X_ab is a linear combo of A_ij elements.
    # Since A_ij is a 2x2 block, X_ab = sum_{i,j} c_{ij} * A_{ij, ab}
    # Actually, X is formed by adding blocks.
    # X = A00 + A11 means X_00 = A00_00 + A11_00, etc.
    
    # Let's define the block coefficients for X and Y for each outer step.
    # x_blocks[outer][a][b] = {block_idx: coeff}
    # Where block_idx is (i,j) in 0..1.
    
    outer_steps = [
        # 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # 3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # 4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # 5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # 7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Inner Strassen steps S1..S7
    # S1 = (X00+X11)(Y00+Y11)
    # S2 = (X10+X11)Y00
    # S3 = X00(Y01-Y11)
    # S4 = X11(Y10-Y00)
    # S5 = (X00+X01)Y11
    # S6 = (X10-X00)(Y00+Y01)
    # S7 = (X01-X11)(Y10+Y11)
    
    # Coefficients of X_ab in each S_s
    # x_s[s][a][b] = coeff
    # Note: X_ab is a scalar value computed from A.
    # So we just need the linear combo of X_ab's.
    
    x_coeffs = [
        # S1
        {(0,0): 1, (1,1): 1},
        # S2
        {(1,0): 1, (1,1): 1},
        # S3
        {(0,0): 1},
        # S4
        {(1,1): 1},
        # S5
        {(0,0): 1, (0,1): 1},
        # S6
        {(1,0): 1, (0,0): -1},
        # S7
        {(0,1): 1, (1,1): -1},
    ]
    
    y_coeffs = [
        # S1
        {(0,0): 1, (1,1): 1},
        # S2
        {(0,0): 1},
        # S3
        {(0,1): 1, (1,1): -1},
        # S4
        {(1,0): 1, (0,0): -1},
        # S5
        {(1,1): 1},
        # S6
        {(0,0): 1, (0,1): 1},
        # S7
        {(1,0): 1, (1,1): 1},
    ]
    
    # M coefficients: M_ab = sum_s c_s S_s
    # m_coeffs[a][b][s] = coeff
    m_coeffs = [
        [ # a=0
            [1, 0, 0, 1, -1, 0, 1], # M00: S1 + S4 - S5 + S7
            [0, 0, 1, 0, 1, 0, 0],  # M01: S3 + S5
        ],
        [ # a=1
            [0, 1, 0, 1, 0, 0, 0],  # M10: S2 + S4
            [1, -1, 1, 0, 0, 1, 0]  # M11: S1 - S2 + S3 + S6
        ]
    ]
    
    # C coefficients: C_cd = sum_o d_o M_o
    # c_coeffs[c][d][o] = coeff
    c_coeffs = [
        [ # c=0
            [1, 0, 0, 1, -1, 0, 1], # C00: M1 + M4 - M5 + M7
            [0, 0, 1, 0, 1, 0, 0],  # C01: M3 + M5
        ],
        [ # c=1
            [0, 1, 0, 1, 0, 0, 0],  # C10: M2 + M4
            [1, -1, 1, 0, 0, 1, 0]  # C11: M1 - M2 + M3 + M6
        ]
    ]
    
    for o in range(7):
        x_blocks, y_blocks = outer_steps[o]
        for s in range(7):
            # Determine A coefficients
            # A_coeff[i][j] for block (i,j) and element (a,b)
            # X_ab = sum_{i,j} x_blocks[i,j] * A_{ij, ab}
            # S_s = sum_{a,b} x_coeffs[s][a,b] * X_ab
            # So A_{ij, ab} coeff = x_blocks[i,j] * x_coeffs[s][a,b]
            
            a_coeffs = {} # (i,j,a,b) -> coeff
            for (i,j), bx in x_blocks.items():
                for (a,b), xc in x_coeffs[s].items():
                    key = (i,j,a,b)
                    a_coeffs[key] = a_coeffs.get(key, 0) + bx * xc
            
            # Determine B coefficients
            b_coeffs = {} # (k,l,c,d) -> coeff
            for (k,l), by in y_blocks.items():
                for (c,d), yc in y_coeffs[s].items():
                    key = (k,l,c,d)
                    b_coeffs[key] = b_coeffs.get(key, 0) + by * yc
            
            # Determine C coefficients
            # C_{cd, ab} gets contribution from M_{o, ab}
            # M_{o, ab} = sum_s m_coeffs[ab][s] S_s
            # C_{cd} = sum_o c_coeffs[cd][o] M_o
            # So C_{cd, ab} coeff = c_coeffs[cd][o] * m_coeffs[ab][s]
            
            w_coeffs = {} # (cd, ab) -> coeff
            for c in range(2):
                for d in range(2):
                    for a in range(2):
                        for b in range(2):
                            coeff = c_coeffs[c][d][o] * m_coeffs[a][b][s]
                            if coeff != 0:
                                key = (c,d,a,b)
                                w_coeffs[key] = w_coeffs.get(key, 0) + coeff
            
            # Populate U, V, W
            # U index: (i, j) -> i*4 + j
            # V index: (k, l) -> k*5 + l
            # W index: (k, i) -> k*4 + i  (Trace convention: W[k*n+i])
            # Note: C index is (row, col) = (i, k) in standard AB.
            # Here C is 4x5.
            # In the 4x4x4 part, C is 4x4.
            # C_{cd, ab} corresponds to row 2c+a, col 2d+b.
            # So i = 2c+a, k = 2d+b.
            # W index is k*4 + i.
            
            for (i,j,a,b), ua in a_coeffs.items():
                if ua != 0:
                    u_idx = i*2 + a # Row of A block?
                    # Wait, A is 4x4.
                    # Block (i,j) is rows 2i..2i+1, cols 2j..2j+1.
                    # Element (a,b) in block is row 2i+a, col 2j+b.
                    # So A index is (2i+a, 2j+b).
                    # U index is (row * m + col) = (2i+a)*4 + (2j+b).
                    u_flat = (2*i + a) * 4 + (2*j + b)
                    U[u_flat][col] += ua
            
            for (k,l,c,d), vb in b_coeffs.items():
                if vb != 0:
                    # B is 4x4 part.
                    # Block (k,l) is rows 2k..2k+1, cols 2l..2l+1.
                    # Element (c,d) is row 2k+c, col 2l+d.
                    # V index is (row * p + col). p=5.
                    # But B is 4x5. The 4x4 part uses cols 0..3.
                    # So col is 2l+d.
                    v_flat = (2*k + c) * 5 + (2*l + d)
                    V[v_flat][col] += vb
            
            for (c,d,a,b), wc in w_coeffs.items():
                if wc != 0:
                    # C index row = 2c+a, col = 2d+b.
                    # W index is col * n + row = (2d+b)*4 + (2c+a).
                    w_flat = (2*d + b) * 4 + (2*c + a)
                    W[w_flat][col] += wc
            
            col += 1
            
    # --- Part 2: Schoolbook 4x4x1 ---
    # 16 terms
    for i in range(4):
        for j in range(4):
            # A_ij * B_j0
            u_flat = i * 4 + j
            v_flat = j * 5 + 0
            w_flat = 0 * 4 + i # k=0, n=4
            
            U[u_flat][col] = 1
            V[v_flat][col] = 1
            W[w_flat][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
