"""Program for 6x6 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns rank-189 decomposition of the 6x6x6 matrix multiplication tensor
    using the Strassen algorithm on 3x3 blocks, each computed by schoolbook (rank 27).
    """
    deadline = time.time() + 100  # safety margin well below 120s

    n, m, p = 6, 6, 6
    # Each 3x3 block product has rank 9 (schoolbook: sum over k of A_ik * B_kj)
    # Wait, let me reconsider. The tensor for n x n matrix multiplication
    # T[i,j, k,l, m,n] = delta_{j,k} * delta_{l,m}
    # The rank is n^3. For n=3, rank = 27.
    # Each term is A[i,k] * B[k,j] * output delta.
    # So for each Strassen product of two 3x3 matrices, we need rank 27.
    # 7 Strassen products * 27 = 189.

    rank = 189
    U = [[0] * rank for _ in range(36)]  # n*m = 36
    V = [[0] * rank for _ in range(36)]  # m*p = 36
    W = [[0] * rank for _ in range(36)]  # p*n = 36

    # Strassen formulas for C = A * B where A, B, C are 3x3:
    # Split into top-left (0,0), top-right (0,1), bottom-left (1,0), bottom-right (1,1)
    # But 3x3 doesn't split evenly into 2x2 blocks.
    # 
    # Alternative approach: Use Strassen on 2x2 blocks where each block is... 
    # No, 3 doesn't split into 2x2.
    #
    # Better approach: Decompose 6x6 as a Kronecker/product structure.
    # 
    # Actually, the cleanest approach for 6x6: 
    # Use the fact that 6 = 2 * 3, and use a mixed decomposition.
    # 
    # Let me think about this differently. The standard way to get rank < 216 for 6x6:
    # - Strassen: rank(2n) = 7 * rank(n). So rank(6) <= 7 * rank(3) = 7 * 27 = 189.
    #   But Strassen works on 2x2 blocks, so it applies when dimension is even.
    #   6 is not 2 * 3 in the Strassen sense (Strassen doubles the size).
    #   
    # - Alternative: View 6x6 multiplication as built from smaller blocks.
    #   If we use 2x2 blocks where each submatrix is 3x3, that's not standard Strassen.
    #
    # Let me use a different decomposition. 
    #
    # Actually, I can use the following: decompose 6x6 multiplication using
    # the tensor rank formula. The rank of n x n x n multiplication tensor is at most n^3.
    # For n=6, that's 216.
    #
    # To beat 216, I can use Strassen-style: 
    # rank(6x6) <= 7 * rank(3x3) if we can apply Strassen.
    # But Strassen requires the matrices to be decomposable into 2x2 blocks.
    #
    # Another approach: Use the Coppersmith-Winograd or other algorithms, but those
    # are complex to implement and may not give integer/half-integer entries.
    #
    # Let me try a simpler approach: use the bilinear decomposition directly.
    # The tensor T[i,j, k,l, m,n] = delta_{j,k} * delta_{l,m} for i in 0..5, k in 0..5, m in 0..5.
    # 
    # Actually, the cleanest way to get rank 189:
    # Think of 6x6 as a block matrix with 2x2 blocks, where each block is 3x3.
    # A = [[A00, A01], [A10, A11]] where each Aij is 3x3.
    # B = [[B00, B01], [B10, B11]] where each Bij is 3x3.
    # C = [[C00, C01], [C10, C11]] where each Cij is 3x3.
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mj is a product of two 3x3 matrices, computed by schoolbook (rank 9).
    # Wait, no. The rank of 3x3 matrix multiplication is 27, not 9.
    # 
    # For 3x3: C[i,j] = sum_k A[i,k] * B[k,j]
    # There are 3*3 = 9 output elements, each is a sum of 3 products.
    # The tensor rank is 27 (one term for each (i,k,j) triple).
    #
    # So each Mj has rank 27. Total rank = 7 * 27 = 189.

    # Index mapping:
    # A is 6x6: row = block_row * 3 + local_row, col = block_col * 3 + local_col
    # B is 6x6: row = block_row * 3 + local_row, col = block_col * 3 + local_col
    # C is 6x6: row = block_row * 3 + local_row, col = block_col * 3 + local_col

    # U index: A[row, col] where row = block_row*3 + local_row, col = block_col*3 + local_col
    # V index: B[row, col] where row = block_row*3 + local_row, col = block_col*3 + local_col
    # W index: C[row, col] where row = block_row*3 + local_row, col = block_col*3 + local_col

    # For each Strassen product Mj = X * Y where X and Y are 3x3:
    #   X is a linear combination of A blocks
    #   Y is a linear combination of B blocks
    #   The product is computed by schoolbook: for each (i, k, j) in 3x3x3:
    #       C_block[i,j] += X[i,k] * Y[k,j]
    #   This contributes to the output C blocks.

    # Let me define the Strassen terms more carefully.
    # Each term is: output_block = sum of (coef * A_block_i * B_block_j) for various block combos.
    
    # For simplicity, let me enumerate all 189 terms.
    # Term index: s * 27 + t where s in 0..6 (Strassen product) and t in 0..26 (schoolbook term within)
    
    # For Strassen product s, we have X_s and Y_s which are 3x3 matrices that are
    # linear combinations of A and B blocks respectively.
    # Then for each (li, lk, lj) in {0,1,2}^3:
    #   X_s[li, lk] is a linear combination of A blocks
    #   Y_s[lk, lj] is a linear combination of B blocks
    #   The product contributes to C_block[output_block][li, lj]
    
    # Let me define for each Strassen product s:
    #   X_blocks[s]: list of (block_row, block_col, coef) for the A-side
    #   Y_blocks[s]: list of (block_row, block_col, coef) for the B-side
    #   But X_s[li, lk] is the same linear combination for all li, lk... 
    #   No, X_s is a 3x3 matrix where each entry is a linear combination of A blocks.
    #   But in Strassen, X_s is formed by adding A blocks, so X_s[li, lk] = sum of coefs * A[br, bc][li, lk]
    
    # Actually, in Strassen, the products are of the form (A_block_i + A_block_j) * (B_block_k + B_block_l)
    # So X_s[li, lk] = A_block_i[li, lk] + A_block_j[li, lk] (for some i, j)
    # And Y_s[lk, lj] = B_block_k[lk, lj] + B_block_l[lk, lj] (for some k, l)
    
    # The product X_s * Y_s = sum_{lk} X_s[:, lk] * Y_s[lk, :]
    # For each (li, lk, lj):
    #   contribution = X_s[li, lk] * Y_s[lk, lj]
    #   = (sum_a coef_a * A[ba_a, ba_ba][li, lk]) * (sum_b coef_b * B[bb_b, bb_bb][lk, lj])
    #   This is a bilinear term: (sum_a coef_a * A[ba_a, ba_ba][li, lk]) * (sum_b coef_b * B[bb_b, bb_bb][lk, lj])
    
    # For the tensor decomposition, each such bilinear term maps to:
    #   U entry: for the A part, set U[A_row, A_col, term_idx] = coef_a for each (a)
    #   Wait, no. Each term in the decomposition is a rank-1 term.
    #   The bilinear form is: (linear combo of A entries) * (linear combo of B entries) * (output entry)
    #   This can be decomposed into multiple rank-1 terms.
    
    # Actually, the standard schoolbook decomposition for 3x3:
    # C[li, lj] = sum_lk A[li, lk] * B[lk, lj]
    # For each (li, lk, lj), we have a rank-1 term:
    #   U[A_index(li, lk), term] = 1
    #   V[B_index(lk, lj), term] = 1
    #   W[C_index(li, lj), term] = 1
    # where A_index, B_index, C_index map to the 6x6 positions.
    
    # But in Strassen, A[li, lk] is replaced by a linear combination of A blocks.
    # So the term becomes:
    #   (sum_a coef_a * A[ba_a, ba_ba][li, lk]) * (sum_b coef_b * B[bb_b, bb_bb][lk, lj]) * output
    #   = sum_a sum_b coef_a * coef_b * A[ba_a, ba_ba][li, lk] * B[bb_b, bb_bb][lk, lj] * output
    
    # Each (a, b) pair gives a rank-1 term. But we can group them.
    # Actually, for the decomposition to work, each term must be truly rank-1.
    # The product of two linear combinations is NOT rank-1 in general.
    
    # So I need to expand: (sum_a coef_a * A_a) * (sum_b coef_b * B_b) = sum_a sum_b coef_a*coef_b * A_a * B_b
    # Each (a, b) pair is a separate rank-1 term.
    
    # This means the rank is not simply 7*27. Let me recount.
    # For each Strassen product, X_s has at most 2 A-block terms, Y_s has at most 2 B-block terms.
    # So X_s[li, lk] = sum of up to 2 terms, Y_s[lk, lj] = sum of up to 2 terms.
    # Product has up to 4 rank-1 terms per (li, lk, lj).
    # 27 * 4 = 108 per Strassen product, 7 * 108 = 756. That's worse than 216!
    
    # I'm making an error. Let me reconsider.
    # 
    # In the tensor decomposition, a rank-1 term is:
    #   u_i * v_j * w_k where u is a vector in A-space, v in B-space, w in C-space.
    # 
    # The schoolbook term for 3x3: A[li, lk] * B[lk, lj] * C_delta[li, lj]
    # Here A[li, lk] is a single entry of A (not a linear combination).
    # 
    # In Strassen, when we write (A00 + A11) * (B00 + B11), this is NOT a single rank-1 term.
    # It's 4 rank-1 terms: A00*B00, A00*B11, A11*B00, A11*B11.
    # 
    # So the rank of the Strassen decomposition for 2x2 blocks of size n is:
    # 7 products, each of which expands to n^2 * (number of A terms) * (number of B terms) rank-1 terms.
    # For the 2x2 Strassen:
    # M1 = (A00+A11)(B00+B11): 4 * n^2 terms
    # M2 = (A10+A11)B00: 2 * n^2 terms
    # M3 = A00(B01-B11): 2 * n^2 terms
    # M4 = A11(B10-B00): 2 * n^2 terms
    # M5 = (A00+A01)B11: 2 * n^2 terms
    # M6 = (A10-A00)(B00+B01): 4 * n^2 terms
    # M7 = (A01-A11)(B10+B11): 4 * n^2 terms
    # Total: (4+2+2+2+2+4+4) * n^2 = 20 * n^2
    # For n=3: 20 * 9 = 180. Plus the additions don't add to rank.
    # Wait, but the additions in C (like C00 = M1 + M4 - M5 + M7) don't add to the tensor rank; they just mean the W entries are summed.
    
    # So total rank = 180? Let me verify: 20 * 9 = 180. That's better than 189!
    # But wait, I think I'm double-counting. Each "term" in the above is actually a rank-1 term for the 3x3 product.
    # For the 3x3 product X*Y where X is 3x3 and Y is 3x3:
    #   If X = X1 + X2 (linear combo of two 3x3 matrices) and Y = Y1 + Y2:
    #   X*Y = X1*Y1 + X1*Y2 + X2*Y1 + X2*Y2
    #   Each X_i * Y_j is a 3x3 matrix product with rank 9 (schoolbook).
    #   So total rank for this Strassen product = 4 * 9 = 36.
    
    # So:
    # M1: 4 * 9 = 36
    # M2: 2 * 9 = 18
    # M3: 2 * 9 = 18
    # M4: 2 * 9 = 18
    # M5: 2 * 9 = 18
    # M6: 4 * 9 = 36
    # M7: 4 * 9 = 36
    # Total: 36+18+18+18+18+36+36 = 180
    
    # Great, so rank = 180.
    
    # Now let me implement this.
    
    rank = 180
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    col = 0
    
    # A block (br, bc) corresponds to A[br*3:br*3+3, bc*3:bc*3+3]
    # A_index(br, bc, li, lk) = (br*3 + li) * 6 + (bc*3 + lk)  -- row in U
    # B block (br, bc) corresponds to B[br*3:br*3+3, bc*3:bc*3+3]
    # B_index(br, bc, lk, lj) = (br*3 + lk) * 6 + (bc*3 + lj)  -- row in V
    # C block (br, bc) corresponds to C[br*3:br*3+3, bc*3:bc*3+3]
    # C_index(br, bc, li, lj) = (br*3 + li) * 6 + (bc*3 + lj)  -- row in W
    
    # For each Strassen product, define the A-side and B-side terms.
    # Each term is (block_row, block_col, coef)
    # X_s = sum of A_terms, Y_s = sum of B_terms
    # For each pair (a_term, b_term) and each (li, lk, lj):
    #   U[A_index(a_term.br, a_term.bc, li, lk), col] += a_term.coef
    #   V[B_index(b_term.br, b_term.bc, lk, lj), col] += b_term.coef
    #   W[C_index(output_br, output_bc, li, lj), col] += 1
    #   col += 1
    
    # Strassen products:
    # M1 = (A00 + A11)(B00 + B11) -> contributes to C00, C01, C10, C11 with coefs
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen product s, for each (a, b) in a_terms x b_terms:
    #   For each (li, lk, lj):
    #     The output C[li, lj] in the output block gets contribution:
    #       a.coef * b.coef * 1
    #     But we need to distribute to the correct C blocks based on the Strassen formula.
    
    # Let me define for each Strassen product s:
    #   a_terms[s]: list of (br, bc, coef)
    #   b_terms[s]: list of (br, bc, coef)
    #   c_contrib[s]: dict mapping (c_br, c_bc) to coef (how M_s contributes to C blocks)
    
    a_terms = {
        0: [(0,0,1), (1,1,1)],  # M1
        1: [(1,0,1), (1,1,1)],  # M2
        2: [(0,0,1)],           # M3
        3: [(1,1,1)],           # M4
        4: [(0,0,1), (0,1,1)],  # M5
        5: [(1,0,1), (0,0,-1)], # M6
        6: [(0,1,1), (1,1,-1)], # M7
    }
    b_terms = {
        0: [(0,0,1), (1,1,1)],  # M1
        1: [(0,0,1)],           # M2
        2: [(0,1,1), (1,1,-1)], # M3
        3: [(1,0,1), (0,0,-1)], # M4
        4: [(1,1,1)],           # M5
        5: [(0,0,1), (0,1,1)],  # M6
        6: [(1,0,1), (1,1,1)],  # M7
    }
    c_contrib = {
        0: {(0,0):1, (0,1):0, (1,0):0, (1,1):1},  # M1 -> C00 + C11
        1: {(0,0):0, (0,1):0, (1,0):1, (1,1):-1},  # M2 -> C10 - C11
        2: {(0,0):0, (0,1):1, (1,0):0, (1,1):1},   # M3 -> C01 + C11
        3: {(0,0):1, (0,1):0, (1,0):1, (1,1):0},   # M4 -> C00 + C10
        4: {(0,0):-1, (0,1):1, (1,0):0, (1,1):0},  # M5 -> -C00 + C01
        5: {(0,0):0, (0,1):0, (1,0):0, (1,1):1},   # M6 -> C11
        6: {(0,0):1, (0,1):0, (1,0):0, (1,1):0},   # M7 -> C00
    }
    
    for s in range(7):
        for (abr, abc, acoef) in a_terms[s]:
            for (bbr, bbc, bcoef) in b_terms[s]:
                for li in range(3):
                    for lk in range(3):
                        for lj in range(3):
                            u_idx = (abr*3 + li) * 6 + (abc*3 + lk)
                            v_idx = (bbr*3 + lk) * 6 + (bbc*3 + lj)
                            for (cbr, cbc), ccoef in c_contrib[s].items():
                                if ccoef != 0:
                                    w_idx = (cbr*3 + li) * 6 + (cbc*3 + lj)
                                    U[u_idx][col] = acoef
                                    V[v_idx][col] = bcoef
                                    W[w_idx][col] = ccoef
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
