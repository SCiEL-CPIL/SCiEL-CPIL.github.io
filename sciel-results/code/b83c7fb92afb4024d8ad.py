"""Matrix multiplication 4x4x4: rank-49 Strassen decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using a two-level Strassen construction.
    
    The output W follows the official trace convention W[k*n+i, r] where
    (k*n+i) is the row index in W, n is the dimension of the first matrix,
    and i is the column index in the result.
    
    U has shape (n*m, rank) = (16, 49)
    V has shape (m*p, rank) = (16, 49)
    W has shape (p*n, rank) = (16, 49)
    """
    n = m = p = 4
    rank = 49
    
    # We build the decomposition by composing two levels of Strassen.
    # 
    # Level 1: Partition 4x4 matrices into 2x2 blocks of size 2x2.
    # This gives 7 "block multiplications", each of which is a 2x2 matrix product.
    # 
    # Level 2: Each 2x2 matrix product is decomposed into 7 scalar multiplications
    # using Strassen's algorithm.
    #
    # Total: 7 * 7 = 49 scalar multiplications.
    #
    # The key idea: each of the 49 scalar multiplications computes
    #   (linear combo of A entries) * (linear combo of B entries)
    # and contributes to specific entries of the result C = A*B.
    #
    # We need to determine, for each of the 49 scalar products:
    #   - Which A entries appear (with what coefficients) -> U vector
    #   - Which B entries appear (with what coefficients) -> V vector  
    #   - Which C entries receive the product (with what coefficients) -> W vector
    #
    # Strassen 2x2 decomposition:
    # For X (2x2) * Y (2x2) = Z (2x2):
    #   M1 = (X00 + X11) * (Y00 + Y11)
    #   M2 = (X10 + X11) * Y00
    #   M3 = X00 * (Y01 - Y11)
    #   M4 = X11 * (Y10 - Y00)
    #   M5 = (X00 + X01) * Y11
    #   M6 = (X10 - X00) * (Y00 + Y01)
    #   M7 = (X01 - X11) * (Y10 + Y11)
    #
    #   Z00 = M1 - M4 + M5 - M7
    #   Z01 = M3 + M7
    #   Z10 = M2 + M4
    #   Z11 = M1 - M2 + M3 + M6
    #
    # Strassen 4x4 (block level), with A, B partitioned into 2x2 blocks:
    #   A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    #   C = A*B where C00 = A00*B00 + A01*B10, etc.
    #
    # Using Strassen at the block level:
    #   P1 = (A00 + A11) * (B00 + B11)
    #   P2 = (A10 + A11) * B00
    #   P3 = A00 * (B01 - B11)
    #   P4 = A11 * (B10 - B00)
    #   P5 = (A00 + A01) * B11
    #   P6 = (A10 - A00) * (B00 + B01)
    #   P7 = (A01 - A11) * (B10 + B11)
    #
    #   C00 = P1 - P4 + P5 - P7
    #   C01 = P3 + P7
    #   C10 = P2 + P4
    #   C11 = P1 - P2 + P3 + P6
    #
    # Now each Pi is itself a 2x2 matrix product, computed with 7 scalar mults.
    
    # Define the Strassen 2x2 scalar decomposition coefficients.
    # For a 2x2 product X*Y where X has entries (x00,x01,x10,x11) indexed 0-3
    # and Y has entries (y00,y01,y10,y11) indexed 0-3:
    #
    # The 7 terms (u, v) pairs where u is a vector over X entries, v over Y entries:
    strassen_2x2 = [
        # (u_coefficients over [x00,x01,x10,x11], v_coefficients over [y00,y01,y10,y11], output_contrib)
        # M1 = (x00+x11)(y00+y11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 1), (3, 1)]),   # z00 += M1, z11 += M1
        # M2 = (x10+x11)y00
        ([0, 0, 1, 1], [1, 0, 0, 0], [(2, 1), (3, -1)]),  # z10 += M2, z11 -= M2
        # M3 = x00(y01-y11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [(1, 1), (3, 1)]),  # z01 += M3, z11 += M3
        # M4 = x11(y10-y00)
        ([0, 0, 0, 1], [0, 0, 1, -1], [(0, -1), (2, 1)]), # z00 -= M4, z10 += M4
        # M5 = (x00+x01)y11
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, 1), (3, 1)]),   # z00 += M5, z11 += M5
        # M6 = (x10-x00)(y00+y01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(3, 1)]),          # z11 += M6
        # M7 = (x01-x11)(y10+y11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, -1), (1, 1)]), # z00 -= M7, z01 += M7
    ]
    
    # Define the Strassen 4x4 block decomposition.
    # Block indices: A00=block(0,0), A01=block(0,1), A10=block(1,0), A11=block(1,1)
    # Each block is a 2x2 matrix.
    # For each of the 7 block products Pi = Xi * Yi:
    #   Xi and Yi are linear combinations of the 4 blocks of A and B respectively.
    #   The result Zi contributes to the 4 blocks of C with certain signs.
    #
    # Block Strassen:
    #   P1: X = A00 + A11, Y = B00 + B11 -> contributes to C00(+1), C11(+1)
    #   P2: X = A10 + A11, Y = B00       -> contributes to C10(+1), C11(-1)
    #   P3: X = A00,         Y = B01 - B11 -> contributes to C01(+1), C11(+1)
    #   P4: X = A11,         Y = B10 - B00 -> contributes to C00(-1), C10(+1)
    #   P5: X = A00 + A01,  Y = B11       -> contributes to C00(+1), C11(+1)
    #   P6: X = A10 - A00,  Y = B00 + B01 -> contributes to C11(+1)
    #   P7: X = A01 - A11,  Y = B10 + B11 -> contributes to C00(-1), C01(+1)
    #
    # For each block product Pi = Xi * Yi, we need to:
    #   1. Express Xi as a linear combination of the 4 A-blocks (each 2x2 = 4 scalar entries)
    #   2. Express Yi as a linear combination of the 4 B-blocks
    #   3. Use the 2x2 Strassen decomposition to get 7 scalar terms
    #   4. For each scalar term, the U-vector is the composition of the Xi-linear-combo
    #      with the 2x2 u-coefficients, similarly for V.
    #   5. The W-vector gets the product of the block-level sign and the 2x2 z-contribution.
    
    # Indexing for 4x4 matrix entries:
    # A is 4x4, row-major: A[i*m+j] where m=4
    # A00 = A[0:2, 0:2] -> entries A[0], A[1], A[4], A[5]  (indices 0,1,4,5)
    # A01 = A[0:2, 2:4] -> entries A[2], A[3], A[6], A[7]  (indices 2,3,6,7)
    # A10 = A[2:4, 0:2] -> entries A[8], A[9], A[12], A[13] (indices 8,9,12,13)
    # A11 = A[2:4, 2:4] -> entries A[10], A[11], A[14], A[15] (indices 10,11,14,15)
    
    # B is 4x4, row-major: B[j*p+k] where p=4
    # B00 = B[0:2, 0:2] -> entries B[0], B[1], B[4], B[5]
    # B01 = B[0:2, 2:4] -> entries B[2], B[3], B[6], B[7]
    # B10 = B[2:4, 0:2] -> entries B[8], B[9], B[12], B[13]
    # B11 = B[2:4, 2:4] -> entries B[10], B[11], B[14], B[15]
    
    # C is 4x4, W index: W[k*n+i, r] where n=4
    # C00 = C[0:2, 0:2] -> W indices: k*n+i for (k,i) in {(0,0),(0,1),(1,0),(1,1)} = {0,1,4,5}
    # C01 = C[0:2, 2:4] -> W indices: {2,3,6,7}
    # C10 = C[2:4, 0:2] -> W indices: {8,9,12,13}
    # C11 = C[2:4, 2:4] -> W indices: {10,11,14,15}
    
    # For a 2x2 block at position (row_block, col_block) in the 4x4 matrix:
    # The 4 entries are at positions:
    #   (row_block*2 + 0)*4 + (col_block*2 + 0)  = row_block*8 + col_block*2
    #   (row_block*2 + 0)*4 + (col_block*2 + 1)  = row_block*8 + col_block*2 + 1
    #   (row_block*2 + 1)*4 + (col_block*2 + 0)  = row_block*8 + col_block*2 + 4
    #   (row_block*2 + 1)*4 + (col_block*2 + 1)  = row_block*8 + col_block*2 + 5
    
    def block_indices(row_block, col_block):
        """Return the 4 indices of a 2x2 block in a 4x4 row-major matrix."""
        base = row_block * 8 + col_block * 2
        return [base, base + 1, base + 4, base + 5]
    
    # A-block indices (U has 16 rows, indexed 0-15)
    A_block_idx = {
        (0, 0): block_indices(0, 0),  # [0,1,4,5]
        (0, 1): block_indices(0, 1),  # [2,3,6,7]
        (1, 0): block_indices(1, 0),  # [8,9,12,13]
        (1, 1): block_indices(1, 1),  # [10,11,14,15]
    }
    
    # B-block indices (V has 16 rows, indexed 0-15)
    B_block_idx = {
        (0, 0): block_indices(0, 0),
        (0, 1): block_indices(0, 1),
        (1, 0): block_indices(1, 0),
        (1, 1): block_indices(1, 1),
    }
    
    # W-block indices (W has 16 rows, indexed 0-15)
    # W index = k*n + i = k*4 + i
    # C-block (row_block, col_block) has entries at:
    #   (row_block*2 + dr)*4 + (col_block*2 + dc) for dr,dc in {0,1}
    W_block_idx = {
        (0, 0): [0*4+0, 0*4+1, 1*4+0, 1*4+1],  # [0,1,4,5]
        (0, 1): [0*4+2, 0*4+3, 1*4+2, 1*4+3],  # [2,3,6,7]
        (1, 0): [2*4+0, 2*4+1, 3*4+0, 3*4+1],  # [8,9,12,13]
        (1, 1): [2*4+2, 2*4+3, 3*4+2, 3*4+3],  # [10,11,14,15]
    }
    
    # Define the 7 block-level Strassen products for 4x4:
    # Each entry: (X_coeffs over 4 A-blocks, Y_coeffs over 4 B-blocks, output_contribs)
    # X_coeffs: dict from block_index to coefficient
    # Y_coeffs: dict from block_index to coefficient
    # output_contribs: list of (C_block_index, sign)
    
    block_strassen_4x4 = [
        # P1: X = A00 + A11, Y = B00 + B11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, [((0,0), 1), ((1,1), 1)]),
        # P2: X = A10 + A11, Y = B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, [((1,0), 1), ((1,1), -1)]),
        # P3: X = A00, Y = B01 - B11
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, [((0,1), 1), ((1,1), 1)]),
        # P4: X = A11, Y = B10 - B00
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, [((0,0), -1), ((1,0), 1)]),
        # P5: X = A00 + A01, Y = B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, [((0,0), 1), ((1,1), 1)]),
        # P6: X = A10 - A00, Y = B00 + B01
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, [((1,1), 1)]),
        # P7: X = A01 - A11, Y = B10 + B11
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, [((0,0), -1), ((0,1), 1)]),
    ]
    
    # Now construct U, V, W matrices
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for bi, (x_coeffs, y_coeffs, out_contribs) in enumerate(block_strassen_4x4):
        # For this block product, use the 2x2 Strassen decomposition
        for si, (u_local, v_local, z_contribs) in enumerate(strassen_2x2):
            # u_local is a 4-vector for the X 2x2 matrix entries
            # v_local is a 4-vector for the Y 2x2 matrix entries
            # We need to compose with the block-level linear combinations
            
            # Build the full U vector (16 entries)
            u_full = [0] * 16
            for (br, bc), coeff in x_coeffs.items():
                idxs = A_block_idx[(br, bc)]
                for li in range(4):
                    u_full[idxs[li]] += coeff * u_local[li]
            
            # Build the full V vector (16 entries)
            v_full = [0] * 16
            for (br, bc), coeff in y_coeffs.items():
                idxs = B_block_idx[(br, bc)]
                for li in range(4):
                    v_full[idxs[li]] += coeff * v_local[li]
            
            # Build the full W vector (16 entries)
            w_full = [0] * 16
            for (cr, cc), z_sign in z_contribs:
                # z_contribs gives contributions to the 2x2 result Z
                # z_contribs entries are (z_index, sign) where z_index is 0-3
                # mapping to Z[0,0]=0, Z[0,1]=1, Z[1,0]=2, Z[1,1]=3
                # The C_block (cr, cc) receives this contribution
                c_block_sign = None
                for (ob, ob_sign) in out_contribs:
                    if ob == (cr, cc):
                        c_block_sign = ob_sign
                        break
                if c_block_sign is None:
                    continue
                total_sign = c_block_sign * z_sign
                idxs = W_block_idx[(cr, cc)]
                # z_contribs gives (z_entry_index, sign)
                # z_entry_index maps to position within the 2x2 block
                # The z_contribs list contains (pos, sign) pairs
                pass
            
            # Let me redo the W construction more carefully
            # z_contribs is a list of (z_pos, sign) where z_pos is 0-3
            # For each output block (ob, ob_sign) in out_contribs:
            #   the Z 2x2 result is added to C_block(ob) with sign ob_sign
            #   So for each (z_pos, z_sign) in z_contribs:
            #     the entry at position z_pos in C_block(ob) gets contribution ob_sign * z_sign
            
            w_full = [0] * 16
            for (ob, ob_sign) in out_contribs:
                idxs = W_block_idx[ob]
                for (z_pos, z_sign) in z_contribs:
                    w_full[idxs[z_pos]] += ob_sign * z_sign
            
            # Store
            for i in range(16):
                if u_full[i] != 0:
                    U[i][col] = u_full[i]
                if v_full[i] != 0:
                    V[i][col] = v_full[i]
                if w_full[i] != 0:
                    W[i][col] = w_full[i]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
