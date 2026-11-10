import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8
    # Rank 56 decomposition based on 4 independent 2x2x2 Strassen multiplications 
    # applied to the block structure of 2x4 by 4x8.
    # Specifically, A (2x4) is split into A_L, A_R (2x2).
    # B (4x8) is split into B_TL, B_TR (top 2 rows) and B_BL, B_BR (bottom 2 rows).
    # C = A_L B_T + A_R B_B.
    # B_T = [B_TL, B_TR], B_B = [B_BL, B_BR].
    # A_L B_T = A_L B_TL + A_L B_TR.
    # A_R B_B = A_R B_BL + A_R B_BR.
    # Each 2x2x2 product has rank 7.
    # Total 4 * 7 = 28 ranks for 2x4x4.
    # We have two such 2x4x4 blocks for the 4x8 width? 
    # Wait, B is 4x8.
    # Split B into B_left (4x4) and B_right (4x4).
    # C = A B_left + A B_right.
    # Each is 2x4x4 multiplication with rank 28.
    # Total rank 56.
    
    rank = 56
    
    # Initialize U, V, W as zero matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to apply a 2x2 Strassen decomposition for a specific block mapping
    # We need to map the logical Strassen products to the global indices.
    
    # The 4 independent 2x2x2 Strassen instances are:
    # 1. A_L * B_TL
    # 2. A_L * B_TR
    # 3. A_R * B_BL
    # 4. A_R * B_BR
    
    # Indices for A (2x4):
    # A_L: rows 0-1, cols 0-1
    # A_R: rows 0-1, cols 2-3
    
    # Indices for B (4x8):
    # B_TL: rows 0-1, cols 0-3 (Wait, B is 4x8. Top 2 rows are 0-1. Left 4 cols are 0-3? No, 4x4 blocks.)
    # Let's clarify B blocks.
    # B is 4x8.
    # B_TL: rows 0-1, cols 0-3
    # B_TR: rows 0-1, cols 4-7
    # B_BL: rows 2-3, cols 0-3
    # B_BR: rows 2-3, cols 4-7
    
    # Strassen for 2x2 * 2x2 -> 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4+M5-M7
    # C01 = M3+M7
    # C10 = M2+M4
    # C11 = M1-M2+M3+M5
    
    # We will iterate through the 4 blocks and fill the U, V, W matrices.
    # Each block uses 7 columns of U, V, W.
    
    block_defs = [
        # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
        # A is 2x4. Row 0-1.
        # Block 1: A_L (cols 0-1), B_TL (rows 0-1, cols 0-3).
        # C contribution goes to C (rows 0-1, cols 0-3).
        (0, 0, 0, 0, 0, 0),
        # Block 2: A_L (cols 0-1), B_TR (rows 0-1, cols 4-7).
        # C contribution goes to C (rows 0-1, cols 4-7).
        (0, 0, 0, 4, 0, 4),
        # Block 3: A_R (cols 2-3), B_BL (rows 2-3, cols 0-3).
        # C contribution goes to C (rows 0-1, cols 0-3).
        (0, 2, 2, 0, 0, 0),
        # Block 4: A_R (cols 2-3), B_BR (rows 2-3, cols 4-7).
        # C contribution goes to C (rows 0-1, cols 4-7).
        (0, 2, 2, 4, 0, 4),
    ]
    
    # Define the Strassen coefficients for the 2x2 case
    # Each product k in 0..6
    # U entries for the 2x2 A block
    # V entries for the 2x2 B block
    # W entries for the 2x2 C block
    
    # Strassen products (L, R, C_contrib)
    # L: 2x2 matrix for A
    # R: 2x2 matrix for B
    # C_contrib: list of 4 (row, col, coeff) for C
    
    strassen_products = [
        {
            'U': [[1, 0], [0, 1]],  # A00+A11
            'V': [[1, 0], [0, 1]],  # B00+B11
            'W': [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)] # M1 contributes to all
        },
        {
            'U': [[0, 0], [1, 1]],  # A10+A11
            'V': [[1, 0], [0, 0]],  # B00
            'W': [(1, 0, 1), (1, 1, -1)] # M2: C10 +=, C11 -=
        },
        {
            'U': [[1, 0], [0, 0]],  # A00
            'V': [[0, 1], [0, -1]],  # B01-B11
            'W': [(0, 1, 1), (1, 1, 1)] # M3: C01 +=, C11 +=
        },
        {
            'U': [[0, 0], [0, 1]],  # A11
            'V': [[-1, 0], [1, 0]],  # B10-B00
            'W': [(0, 0, 1), (1, 0, 1)] # M4: C00 +=, C10 +=
        },
        {
            'U': [[1, 1], [0, 0]],  # A00+A01
            'V': [[0, 0], [0, 1]],  # B11
            'W': [(0, 0, -1), (0, 1, 1)] # M5: C00 -=, C01 +=
        },
        {
            'U': [[-1, 0], [1, 0]],  # A10-A00
            'V': [[1, 1], [0, 0]],  # B00+B01
            'W': [(1, 1, 1)] # M6: C11 +=
        },
        {
            'U': [[0, 1], [0, -1]],  # A01-A11
            'V': [[1, 0], [1, 1]],  # B10+B11
            'W': [(0, 0, 1)] # M7: C00 +=
        }
    ]
    
    # Verify W coefficients for standard Strassen 2x2:
    # C00 = M1 + M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5 + M6
    
    # My definitions:
    # M1: W(0,0)=1, W(1,0)=1, W(0,1)=1, W(1,1)=1. Correct for M1 in C00,C10,C01,C11.
    # M2: W(1,0)=1, W(1,1)=-1. Correct.
    # M3: W(0,1)=1, W(1,1)=1. Correct.
    # M4: W(0,0)=1, W(1,0)=1. Correct.
    # M5: W(0,0)=-1, W(0,1)=1. Correct.
    # M6: W(1,1)=1. Correct.
    # M7: W(0,0)=1. Correct.
    
    # Check sums:
    # C00: M1(1) + M4(1) + M5(-1) + M7(1) = 2? 
    # Standard: C00 = M1 + M4 + M5 - M7.
    # My M7 W(0,0) is 1. Should be -1?
    # Let's re-verify Strassen formulas.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5 + M6
    
    # My M7 definition:
    # U: A01-A11. V: B10+B11.
    # C01 needs +M7.
    # C00 needs -M7.
    # My M7 W list: [(0, 0, 1)]. It only adds to C00. It misses C01.
    # And C00 coeff should be -1.
    
    # Let's fix M7:
    # W: [(0, 0, -1), (0, 1, 1)]
    
    # Let's fix M5:
    # C00 needs +M5? Formula: C00 = M1 + M4 + M5 - M7.
    # My M5 W: [(0, 0, -1), (0, 1, 1)].
    # C01 needs +M5? Formula: C01 = M3 + M7. No M5 in C01.
    # Wait.
    # C11 = M1 - M2 + M3 + M5 + M6.
    # So M5 contributes to C11 (+1) and C00 (+1)?
    # Let's re-read standard Strassen.
    # C00 = M1 + M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5 + M6
    
    # So M5 contributes to C00 (+1) and C11 (+1).
    # My M5 W: [(0, 0, -1), (0, 1, 1)]. This is wrong.
    # M5 U: (A00+A01)B11.
    # M5 V: B11.
    # M5 should add to C00 and C11.
    
    # Let's rewrite the Strassen products carefully.
    
    strassen_products = [
        # M1
        {'U': [[1, 0], [0, 1]], 'V': [[1, 0], [0, 1]], 'W': [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]},
        # M2
        {'U': [[0, 0], [1, 1]], 'V': [[1, 0], [0, 0]], 'W': [(1, 0, 1), (1, 1, -1)]},
        # M3
        {'U': [[1, 0], [0, 0]], 'V': [[0, 1], [0, -1]], 'W': [(0, 1, 1), (1, 1, 1)]},
        # M4
        {'U': [[0, 0], [0, 1]], 'V': [[-1, 0], [1, 0]], 'W': [(0, 0, 1), (1, 0, 1)]},
        # M5
        {'U': [[1, 1], [0, 0]], 'V': [[0, 0], [0, 1]], 'W': [(0, 0, 1), (1, 1, 1)]},
        # M6
        {'U': [[-1, 0], [1, 0]], 'V': [[1, 1], [0, 0]], 'W': [(1, 1, 1)]},
        # M7
        {'U': [[0, 1], [0, -1]], 'V': [[1, 0], [1, 1]], 'W': [(0, 0, -1), (0, 1, 1)]},
    ]
    
    col = 0
    for a_r_off, a_c_off, b_r_off, b_c_off, c_r_off, c_c_off in block_defs:
        for prod in strassen_products:
            # Apply this Strassen product to the specific block
            # U (2x4 x 7)
            for u_row in range(2):
                for u_col in range(2):
                    coeff = prod['U'][u_row][u_col]
                    if coeff != 0:
                        U[a_r_off + u_row, a_c_off + u_col] += coeff * 0 # Wait, U is global.
                        # U index: row = a_r_off + u_row, col = a_c_off + u_col
                        # But U is (8, 56).
                        # We are filling column `col`.
                        U[a_r_off + u_row, a_c_off + u_col] # This is wrong. U is (n*m, rank).
                        # U[row, col] += coeff
                        U[(a_r_off + u_row) * m + (a_c_off + u_col), col] += coeff
            
            # V (4x8 x 7)
            for v_row in range(2):
                for v_col in range(2):
                    coeff = prod['V'][v_row][v_col]
                    if coeff != 0:
                        V[(b_r_off + v_row) * p + (b_c_off + v_col), col] += coeff
            
            # W (8x2 x 7)
            for c_r, c_c, coeff in prod['W']:
                if coeff != 0:
                    W[(c_c_off + c_c) * n + (c_r_off + c_r), col] += coeff
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
