import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9

    # We will use a decomposition of rank 114.
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    # C = A B1 + A B2 + A B3.
    # Rank(A B1) <= 49 (Strassen 4x4x4)
    # Rank(A B2) <= 49 (Strassen 4x4x4)
    # Rank(A B3) <= 16 (Schoolbook 4x4x1)
    # Total Rank = 114.

    total_rank = 49 + 49 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Helper to add a 4x4x4 Strassen decomposition
    # offset: column offset in the global rank
    # b_col_offset: column offset in B (0, 4, or 8)
    def add_strassen_4x4x4(offset, b_col_offset, c_col_offset):
        # Strassen 4x4x4: 49 terms.
        # We decompose the product of two 4x4 matrices into 49 rank-1 terms.
        # This is done by viewing the 4x4 matrices as 2x2 blocks of 2x2 matrices.
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # C = A B = [[C00, C01], [C10, C11]]
        
        # Strassen formulas:
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)B00
        # S3 = A00(B01-B11)
        # S4 = A11(B10-B00)
        # S5 = (A00+A01)B11
        # S6 = (A10-A00)(B00+B01)
        # S7 = (A01-A11)(B10+B11)
        
        # C00 = S1 - S4 + S6 - S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S3 + S2 - S5
        
        # Each Si is a product of two 2x2 matrices.
        # We decompose each 2x2 product using Strassen 2x2 (7 terms).
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # P00 = M1 - M4 + M6 - M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M3 + M2 - M5
        
        # We need to map the 2x2 blocks to global indices.
        # A block (r, c) corresponds to A[2r:2r+2, 2c:2c+2]
        # B block (r, c) corresponds to B[2r:2r+2, b_col_offset + 2c : b_col_offset + 2c + 2]
        # C block (r, c) corresponds to C[2r:2r+2, c_col_offset + 2c : c_col_offset + 2c + 2]
        
        # Helper to add a rank-1 term from a 2x2 product decomposition
        # x_indices: list of (row, col, coeff) for X (2x2 matrix of A blocks)
        # y_indices: list of (row, col, coeff) for Y (2x2 matrix of B blocks)
        # c_block: (r, c) block index in C
        def add_2x2_term(x_indices, y_indices, c_block, term_col):
            # x_indices and y_indices are 2x2 matrices of (block_row, block_col, coeff)
            # For each of the 7 Strassen 2x2 terms:
            # M1: X = [[1,0],[0,1]] (X00+X11), Y = [[1,0],[0,1]] (Y00+Y11)
            # M2: X = [[0,0],[1,1]] (X10+X11), Y = [[1,0],[0,0]] (Y00)
            # M3: X = [[1,0],[0,0]] (X00), Y = [[0,1],[0,-1]] (Y01-Y11)
            # M4: X = [[0,0],[0,1]] (X11), Y = [[0,0],[1,-1]] (Y10-Y00)
            # M5: X = [[1,1],[0,0]] (X00+X01), Y = [[0,0],[0,1]] (Y11)
            # M6: X = [[-1,0],[1,0]] (X10-X00), Y = [[1,1],[0,0]] (Y00+Y01)
            # M7: X = [[0,1],[0,-1]] (X01-X11), Y = [[0,0],[1,1]] (Y10+Y11)
            
            strassen_2x2_terms = [
                # (x_coeffs, y_coeffs, c_coeffs)
                # c_coeffs is a 2x2 matrix for the C block
                # M1: C00 += 1, C01 += 0, C10 += 0, C11 += 1
                ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
                # M2: C00 += 0, C01 += 0, C10 += 1, C11 += 0
                ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
                # M3: C00 += 0, C01 += 1, C10 += 0, C11 += 0
                ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 0]]),
                # M4: C00 += -1, C01 += 0, C10 += 1, C11 += 0
                ([[0, 0], [0, 1]], [[0, 0], [1, -1]], [[-1, 0], [1, 0]]),
                # M5: C00 += 0, C01 += 1, C10 += 0, C11 += -1
                ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[0, 1], [0, -1]]),
                # M6: C00 += 1, C01 += 0, C10 += 0, C11 += 0
                ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[1, 0], [0, 0]]),
                # M7: C00 += -1, C01 += 0, C10 += 0, C11 += 0
                ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[-1, 0], [0, 0]]),
            ]
            
            for x_coeffs, y_coeffs, c_coeffs in strassen_2x2_terms:
                # For each entry in the 2x2 coefficient matrices:
                # x_coeffs[i][j] is the coefficient for X[i][j]
                # X[i][j] is a 2x2 block of A
                # Similarly for Y and C.
                
                # Build the U, V, W vectors for this rank-1 term
                u_vec = [0] * 16
                v_vec = [0] * 16
                w_vec = [0] * 16
                
                # X is a 2x2 matrix of A blocks
                # x_indices[i][j] = (block_row, block_col, coeff)
                # The actual A block is at (block_row, block_col)
                # x_coeffs[i][j] is the coefficient for this block
                
                for i in range(2):
                    for j in range(2):
                        if x_coeffs[i][j] != 0:
                            a_row, a_col, a_coeff = x_indices[i][j]
                            # A block (a_row, a_col) is at A[2*a_row:2*a_row+2, 2*a_col:2*a_col+2]
                            for dr in range(2):
                                for dc in range(2):
                                    a_idx = (2 * a_row + dr) * 4 + (2 * a_col + dc)
                                    u_vec[a_idx] += x_coeffs[i][j] * a_coeff
                            
                for i in range(2):
                    for j in range(2):
                        if y_coeffs[i][j] != 0:
                            b_row, b_col, b_coeff = y_indices[i][j]
                            # B block (b_row, b_col) is at B[2*b_row:2*b_row+2, b_col_offset + 2*b_col : b_col_offset + 2*b_col + 2]
                            for dr in range(2):
                                for dc in range(2):
                                    b_idx = (2 * b_row + dr) * 9 + (b_col_offset + 2 * b_col + dc)
                                    v_vec[b_idx] += y_coeffs[i][j] * b_coeff
                            
                for i in range(2):
                    for j in range(2):
                        if c_coeffs[i][j] != 0:
                            # C block (i, j) is at C[2*i:2*i+2, c_col_offset + 2*j : c_col_offset + 2*j + 2]
                            for dr in range(2):
                                for dc in range(2):
                                    c_idx = (2 * i + dr) * 9 + (c_col_offset + 2 * j + dc)
                                    w_vec[c_idx] += c_coeffs[i][j]
                            
                # Add to global U, V, W
                for idx in range(16):
                    if u_vec[idx] != 0:
                        U[idx][term_col] += u_vec[idx]
                    if v_vec[idx] != 0:
                        V[idx][term_col] += v_vec[idx]
                    if w_vec[idx] != 0:
                        W[idx][term_col] += w_vec[idx]
        
        # Now define the 7 Strassen 4x4 terms
        # S1 = (A00+A11)(B00+B11)
        # X1 = A00+A11: x_indices = [[(0,0,1), (0,1,0)], [(1,0,0), (1,1,1)]]
        # Y1 = B00+B11: y_indices = [[(0,0,1), (0,1,0)], [(1,0,0), (1,1,1)]]
        # C00 += S1, C11 += S1
        x1 = [[(0,0,1), (0,1,0)], [(1,0,0), (1,1,1)]]
        y1 = [[(0,0,1), (0,1,0)], [(1,0,0), (1,1,1)]]
        # S1 contributes to C00 and C11
        # C00: add_2x2_term(x1, y1, (0,0), col)
        # C11: add_2x2_term(x1, y1, (1,1), col)
        # But wait, add_2x2_term adds the full 2x2 C block.
        # For S1, we add it to C00 and C11.
        # So we call add_2x2_term twice? No, that would double count.
        # Actually, S1 is a 2x2 matrix. It contributes to all 4 C blocks.
        # C00 += S1[0,0], C01 += S1[0,1], C10 += S1[1,0], C11 += S1[1,1]
        # So we just call add_2x2_term once, and it adds to all 4 C blocks.
        # But in the Strassen formula, S1 is added to C00 and C11 with +1, and to C01 and C10 with 0.
        # Wait, no. S1 is a 2x2 matrix. The formula is:
        # C00 = S1 - S4 + S6 - S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S3 + S2 - S5
        # So S1 contributes to C00 (top-left of S1) and C11 (bottom-right of S1).
        # S1[0,0] goes to C00, S1[1,1] goes to C11.
        # S1[0,1] and S1[1,0] do NOT contribute to C.
        # So we need to modify add_2x2_term to only add specific entries.
        
        # Let's redefine: instead of adding the full 2x2 C block, we specify which C block to add to.
        # But each Si is a 2x2 matrix, and only specific entries go to specific C blocks.
        # Actually, the Strassen formula says:
        # C00 = S1[0,0] - S4[0,0] + S6[0,0] - S7[0,0]
        # C01 = S3[0,0] + S5[0,0]  # Wait, S3 and S5 are 2x2 matrices.
        # No, that's not right. Let me re-derive.
        
        # Standard Strassen 4x4:
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)B00
        # S3 = A00(B01-B11)
        # S4 = A11(B10-B00)
        # S5 = (A00+A01)B11
        # S6 = (A10-A00)(B00+B01)
        # S7 = (A01-A11)(B10+B11)
        
        # C00 = S1 - S4 + S6 - S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S3 + S2 - S5
        
        # Here, C00, C01, C10, C11 are 2x2 matrices.
        # S1, S2, ..., S7 are also 2x2 matrices.
        # So C00[0,0] = S1[0,0] - S4[0,0] + S6[0,0] - S7[0,0]
        # C00[0,1] = S1[0,1] - S4[0,1] + S6[0,1] - S7[0,1]
        # etc.
        
        # So each Si contributes to all 4 C blocks, but with different coefficients.
        # Specifically:
        # S1 contributes to C00 (all entries) and C11 (all entries)
        # S2 contributes to C10 (all entries) and C11 (all entries)
        # S3 contributes to C01 (all entries) and C11 (all entries)
        # S4 contributes to C00 (all entries) and C10 (all entries)
        # S5 contributes to C01 (all entries) and C11 (all entries)
        # S6 contributes to C00 (all entries)
        # S7 contributes to C00 (all entries)
        
        # Wait, let me check:
        # C00 = S1 - S4 + S6 - S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S3 + S2 - S5
        
        # So:
        # S1: C00 (+1), C11 (+1)
        # S2: C10 (+1), C11 (+1)
        # S3: C01 (+1), C11 (-1)
        # S4: C00 (-1), C10 (+1)
        # S5: C01 (+1), C11 (-1)
        # S6: C00 (+1)
        # S7: C00 (-1)
        
        # So for each Si, we need to add its 2x2 matrix to specific C blocks.
        # Let's modify add_2x2_term to take a list of (c_block, coeff) pairs.
        
        # Actually, it's simpler to just call add_2x2_term for each C block separately.
        # But that would require 7 * 4 = 28 calls per Strassen 4x4, which is fine.
        
        # Let's redefine: add_2x2_term(x_indices, y_indices, c_block, coeff, term_col)
        # This adds coeff * (X*Y) to the C block at c_block.
        
        # But X*Y is a 2x2 matrix, and c_block is a 2x2 block of C.
        # So we add the full 2x2 matrix.
        
        # Let's just implement it as is, and call it for each C block.
        
        # For S1: add to C00 with +1, add to C11 with +1
        # For S2: add to C10 with +1, add to C11 with +1
        # etc.
        
        # Let's define the 7 Strassen 4x4 terms and their C block contributions.
        
        strassen_4x4_terms = [
            # (x_indices, y_indices, [(c_block, coeff), ...])
            # S1
            (
                [[(0,0,1), (0,1,0)], [(1,0,0), (1,1,1)]],
                [[(0,0,1), (0,1,0)], [(1,0,0), (1,1,1)]],
                [((0,0), 1), ((1,1), 1)]
            ),
            # S2
            (
                [[(0,0,0), (0,1,0)], [(1,0,1), (1,1,1)]],
                [[(0,0,1), (0,1,0)], [(1,0,0), (1,1,0)]],
                [((1,0), 1), ((1,1), 1)]
            ),
            # S3
            (
                [[(0,0,1), (0,1,0)], [(1,0,0), (1,1,0)]],
                [[(0,0,0), (0,1,1)], [(1,0,0), (1,1,-1)]],
                [((0,1), 1), ((1,1), -1)]
            ),
            # S4
            (
                [[(0,0,0), (0,1,0)], [(1,0,0), (1,1,1)]],
                [[(0,0,0), (0,1,0)], [(1,0,1), (1,1,-1)]],
                [((0,0), -1), ((1,0), 1)]
            ),
            # S5
            (
                [[(0,0,1), (0,1,1)], [(1,0,0), (1,1,0)]],
                [[(0,0,0), (0,1,0)], [(1,0,0), (1,1,1)]],
                [((0,1), 1), ((1,1), -1)]
            ),
            # S6
            (
                [[(0,0,-1), (0,1,0)], [(1,0,1), (1,1,0)]],
                [[(0,0,1), (0,1,1)], [(1,0,0), (1,1,0)]],
                [((0,0), 1)]
            ),
            # S7
            (
                [[(0,0,0), (0,1,1)], [(1,0,0), (1,1,-1)]],
                [[(0,0,0), (0,1,0)], [(1,0,1), (1,1,1)]],
                [((0,0), -1)]
            ),
        ]
        
        for x_indices, y_indices, c_contribs in strassen_4x4_terms:
            for c_block, coeff in c_contribs:
                # Add this term to the global U, V, W
                # We need to add the 2x2 product (X*Y) to the C block at c_block
                # with coefficient coeff.
                
                # Use the Strassen 2x2 decomposition
                strassen_2x2_terms = [
                    ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
                    ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
                    ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 0]]),
                    ([[0, 0], [0, 1]], [[0, 0], [1, -1]], [[-1, 0], [1, 0]]),
                    ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[0, 1], [0, -1]]),
                    ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[1, 0], [0, 0]]),
                    ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[-1, 0], [0, 0]]),
                ]
                
                for x_coeffs, y_coeffs, c_coeffs in strassen_2x2_terms:
                    u_vec = [0] * 16
                    v_vec = [0] * 16
                    w_vec = [0] * 16
                    
                    for i in range(2):
                        for j in range(2):
                            if x_coeffs[i][j] != 0:
                                a_row, a_col, a_coeff = x_indices[i][j]
                                for dr in range(2):
                                    for dc in range(2):
                                        a_idx = (2 * a_row + dr) * 4 + (2 * a_col + dc)
                                        u_vec[a_idx] += x_coeffs[i][j] * a_coeff
                            
                    for i in range(2):
                        for j in range(2):
                            if y_coeffs[i][j] != 0:
                                b_row, b_col, b_coeff = y_indices[i][j]
                                for dr in range(2):
                                    for dc in range(2):
                                        b_idx = (2 * b_row + dr) * 9 + (b_col_offset + 2 * b_col + dc)
                                        v_vec[b_idx] += y_coeffs[i][j] * b_coeff
                            
                    for i in range(2):
                        for j in range(2):
                            if c_coeffs[i][j] != 0:
                                # C block (c_block[0], c_block[1]) is at C[2*c_block[0]:2*c_block[0]+2, c_col_offset + 2*c_block[1] : c_col_offset + 2*c_block[1] + 2]
                                for dr in range(2):
                                    for dc in range(2):
                                        c_idx = (2 * c_block[0] + i + dr) * 9 + (c_col_offset + 2 * c_block[1] + j + dc)
                                        w_vec[c_idx] += coeff * c_coeffs[i][j]
                                        
                    for idx in range(16):
                        if u_vec[idx] != 0:
                            U[idx][term_col] += u_vec[idx]
                        if v_vec[idx] != 0:
                            V[idx][term_col] += v_vec[idx]
                        if w_vec[idx] != 0:
                            W[idx][term_col] += w_vec[idx]
        
        # Check time
        if time.time() - start_time > 100:
            return False
        return True

    # Add Strassen 4x4x4 for B1 (columns 0-3 of B, columns 0-3 of C)
    if not add_strassen_4x4x4(0, 0, 0):
        # Timeout, return what we have so far
        return {"U": U, "V": V, "W": W}
    
    # Add Strassen 4x4x4 for B2 (columns 4-7 of B, columns 4-7 of C)
    if not add_strassen_4x4x4(49, 4, 4):
        return {"U": U, "V": V, "W": W}
    
    # Add schoolbook 4x4x1 for B3 (column 8 of B, column 8 of C)
    # C[:, 8] = A @ B[:, 8]
    # For each i, j: C[i, 8] += A[i, j] * B[j, 8]
    # Rank-1 terms: for each (i, j)
    offset = 98
    for i in range(4):
        for j in range(4):
            col = offset + i * 4 + j
            U[i * 4 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 4 + i][col] = 1
    
    elapsed = time.time() - start_time
    if elapsed > 110:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
