   def _get_strassen_4x4_recursive():
       # 2x2 Strassen factors
       U2, V2, W2 = _get_strassen_2x2()
       
       # Outer block definitions
       # A blocks: 00, 01, 10, 11
       # B blocks: 00, 01, 10, 11
       # C blocks: 00, 01, 10, 11
       
       # Coefficients for X_t (A-side) and Y_t (B-side)
       # Each is a 4x4 matrix of coefficients (for the 4 blocks)
       # But we need to map to 16 elements.
       
       # Let's define the block indices for A (4x4)
       # Block 00: rows 0,1 cols 0,1
       # Block 01: rows 0,1 cols 2,3
       # Block 10: rows 2,3 cols 0,1
       # Block 11: rows 2,3 cols 2,3
       
       # For each outer term t, we have a linear combination of blocks.
       # We represent this as a 4x4 matrix of block coefficients?
       # No, just a list of (block_idx, coeff).
       
       outer_A = [
            [(0, 1), (3, 1)], # M1: A00 + A11
            [(2, 1), (3, 1)], # M2: A10 + A11
            [(0, 1)],         # M3: A00
            [(3, 1)],         # M4: A11
            [(0, 1), (1, 1)], # M5: A00 + A01
            [(2, 1), (0, -1)],# M6: A10 - A00
            [(1, 1), (3, -1)] # M7: A01 - A11
       ]
       
       outer_B = [
            [(0, 1), (3, 1)], # M1: B00 + B11
            [(0, 1)],         # M2: B00
            [(1, 1), (3, -1)],# M3: B01 - B11
            [(2, 1), (0, -1)],# M4: B10 - B00
            [(3, 1)],         # M5: B11
            [(0, 1), (1, 1)], # M6: B00 + B01
            [(2, 1), (3, 1)]  # M7: B10 + B11
       ]
       
       outer_C = [
            [(0, 1), (1, -1), (2, -1), (3, 1)], # C00 = M1 - M4 - M5 + M7
            [(2, 1), (4, 1)],                    # C01 = M3 + M5
            [(1, 1), (3, 1)],                    # C10 = M2 + M4
            [(0, 1), (1, -1), (2, 1), (5, -1)]  # C11 = M1 - M2 + M3 - M6
       ]
       # Wait, outer_C is not a list of 7. It's the coefficients for C blocks in terms of M_t.
       # C_block = sum_t coeff[t] * M_t.
       
       U4 = np.zeros((16, 49))
       V4 = np.zeros((16, 49))
       W4 = np.zeros((16, 49))
       
       for t in range(7):
           for s in range(7):
               col = t * 7 + s
               
               # Build U4 column
               # U4 is 4x4 matrix of coeffs for A.
               # For each block (bi, bj) in A:
               #   coeff_block = outer_A[t][block_idx]
               #   For each element (i, j) in the 2x2 block:
               #     coeff_elem = coeff_block * U2[local_i, local_j, s]
               
               # Map block (bi, bj) to global rows/cols
               # bi, bj in 0,1.
               # Global row = 2*bi + local_i
               # Global col = 2*bj + local_j
               
               for block_idx, coeff_block in outer_A[t]:
                   bi = block_idx // 2
                   bj = block_idx % 2
                   for li in range(2):
                       for lj in range(2):
                           gi = 2 * bi + li
                           gj = 2 * bj + lj
                           U4[gi * 4 + gj, col] += coeff_block * U2[li * 2 + lj, s]
               
               # Build V4 column
               for block_idx, coeff_block in outer_B[t]:
                   bi = block_idx // 2
                   bj = block_idx % 2
                   for li in range(2):
                       for lj in range(2):
                           gi = 2 * bi + li
                           gj = 2 * bj + lj
                           V4[gi * 4 + gj, col] += coeff_block * V2[li * 2 + lj, s]
               
               # Build W4 column
               # W4 is 4x4 matrix of coeffs for C.
               # C_block = sum_t c_t M_t.
               # So for a given C block (ci, cj), the coeff for M_t is outer_C_block[ci, cj][t].
               # Wait, I defined outer_C as a list of 4 items (C00, C01, C10, C11).
               # Each item is a list of (t, coeff).
               
               # For each C block (ci, cj):
               #   Find the coeff for M_t.
               #   For each element (i, j) in the 2x2 block:
               #     coeff_elem = coeff_t * W2[local_i, local_j, s]
               
               for ci in range(2):
                   for cj in range(2):
                       # Get coeff for M_t in C_block(ci, cj)
                       coeff_t = 0
                       for idx, c in outer_C[ci * 2 + cj]:
                           if idx == t:
                               coeff_t = c
                               break
                               
                       if coeff_t != 0:
                           bi = ci
                           bj = cj
                           for li in range(2):
                               for lj in range(2):
                                   gi = 2 * bi + li
                                   gj = 2 * bj + lj
                                   W4[gi * 4 + gj, col] += coeff_t * W2[li * 2 + lj, s]
       
       return U4, V4, W4
   