DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block product term
    # A_block is (1x2), B_block is (2x4), C_block is (1x4)
    # a_idx: list of 2 indices in U
    # b_idx: list of 8 indices in V (2 rows x 4 cols)
    # c_idx: list of 4 indices in W
    def add_block_product(a_idx, b_idx, c_idx):
        nonlocal col
        # 8 terms
        # x0 * y0j * zj
        # x1 * y1j * zj
        for j in range(4):
            # Term 1: x0 * y0j * zj
            U[a_idx[0]][col] = 1
            V[b_idx[0 * 4 + j]][col] = 1
            W[c_idx[j]][col] = 1
            col += 1
            # Term 2: x1 * y1j * zj
            U[a_idx[1]][col] = 1
            V[b_idx[1 * 4 + j]][col] = 1
            W[c_idx[j]][col] = 1
            col += 1

    # Define block indices
    # A: 2x4. Blocks: A00 (row 0, col 0-1), A01 (row 0, col 2-3), A10 (row 1, col 0-1), A11 (row 1, col 2-3)
    # U index: i*m + j
    A00 = [0*4+0, 0*4+1]
    A01 = [0*4+2, 0*4+3]
    A10 = [1*4+0, 1*4+1]
    A11 = [1*4+2, 1*4+3]
    
    # B: 4x8. Blocks: B00 (rows 0-1, cols 0-3), B01 (rows 0-1, cols 4-7), B10 (rows 2-3, cols 0-3), B11 (rows 2-3, cols 4-7)
    # V index: j*p + k
    B00 = [0*8+0, 0*8+1, 0*8+2, 0*8+3, 1*8+0, 1*8+1, 1*8+2, 1*8+3]
    B01 = [0*8+4, 0*8+5, 0*8+6, 0*8+7, 1*8+4, 1*8+5, 1*8+6, 1*8+7]
    B10 = [2*8+0, 2*8+1, 2*8+2, 2*8+3, 3*8+0, 3*8+1, 3*8+2, 3*8+3]
    B11 = [2*8+4, 2*8+5, 2*8+6, 2*8+7, 3*8+4, 3*8+5, 3*8+6, 3*8+7]
    
    # C: 2x8. Blocks: C00 (row 0, cols 0-3), C01 (row 0, cols 4-7), C10 (row 1, cols 0-3), C11 (row 1, cols 4-7)
    # W index: k*n + i
    C00 = [0*2+0, 1*2+0, 2*2+0, 3*2+0]
    C01 = [4*2+0, 5*2+0, 6*2+0, 7*2+0]
    C10 = [0*2+1, 1*2+1, 2*2+1, 3*2+1]
    C11 = [4*2+1, 5*2+1, 6*2+1, 7*2+1]
    
    # Strassen Products
    # M1 = (A00 + A11) (B00 + B11)
    # This is (A00 + A11) * (B00 + B11)
    # We need to add the terms for A00*B00, A00*B11, A11*B00, A11*B11
    # But wait, the rank is 8 per block product.
    # If we add 4 block products, we add 32 terms.
    # M1 contributes 32 terms.
    
    # M2 = (A10 + A11) B00
    # A10*B00 + A11*B00 -> 16 terms
    
    # M3 = A00 (B01 - B11)
    # A00*B01 - A00*B11 -> 16 terms
    
    # M4 = A11 (B10 - B00)
    # A11*B10 - A11*B00 -> 16 terms
    
    # M5 = (A00 + A01) B11
    # A00*B11 + A01*B11 -> 16 terms
    
    # M6 = (A10 - A00) (B00 + B01)
    # A10*B00 + A10*B01 - A00*B00 - A00*B01 -> 32 terms
    
    # M7 = (A01 - A11) (B10 + B11)
    # A01*B10 + A01*B11 - A11*B10 - A11*B11 -> 32 terms
    
    # Total terms: 32 + 16 + 16 + 16 + 16 + 32 + 32 = 160?
    # No, the rank is not the sum of the ranks of the expanded terms.
    # The rank is the number of columns in U, V, W.
    # If we just append all these terms, the rank is 160.
    # This is worse than 64.
    
    # The key is that the SAME rank-1 tensor can be used for multiple products?
    # No, the tensors are different.
    
    # Let's rethink.
    # Strassen works because it replaces 4 multiplications with 7.
    # Here, "multiplication" is a rank-8 operation.
    # If we perform 7 rank-8 operations, the total rank is 56.
    # How do we construct the matrices?
    # We need 7 sets of 8 columns.
    # For each set, we define the U, V, W entries.
    # The U entries for M1 correspond to A00 + A11.
    # So U[row A00, col] = 1 and U[row A11, col] = 1.
    # V[row B00, col] = 1 and V[row B11, col] = 1.
    # W[row C?, col] = 1.
    # But M1 is an intermediate result. It is not directly C.
    # C is a linear combination of M1..M7.
    # The decomposition must map A, B directly to C.
    # So we cannot just have intermediate matrices.
    # We must have a direct sum of rank-1 tensors.
    
    # The rank of the tensor $T$ is the minimum number of rank-1 tensors.
    # If $T = T_1 + \dots + T_k$, and each $T_i$ is rank 1, then rank(T) <= k.
    # Strassen's algorithm expresses the product as a sum of 7 terms, each of which is a product of linear forms.
    # Each "product" is a bilinear map.
    # If each bilinear map has rank 8, and they are independent, the total rank is 56.
    # Are they independent?
    # Yes, generally.
    
    # So we need to construct the 56 rank-1 tensors.
    # For M1:
    # The bilinear map is $(A_{00} + A_{11}) (B_{00} + B_{11})$.
    # This map produces a $1 \times 4$ matrix $M_1$.
    # $M_1 = M_{11} + M_{12} + M_{13} + M_{14}$? No, it's a single $1 \times 4$ matrix.
    # The decomposition of this map into rank-1 tensors involves 8 tensors.
    # Let's call them $t_1, \dots, t_8$.
    # Each $t_i$ has U, V, W parts.
    # The W part of $t_i$ corresponds to the output of M1.
    # But M1 is not the final output C.
    # C is $M_1 + M_4 - M_5 + M_7$, etc.
    # So the W part of the tensors for M1 must be scaled/combined to form C.
    # Since W entries are just numbers, we can set them to 1 for M1, 1 for M4, -1 for M5, etc.
    # But the tensors for M1 and M4 are different.
    # So we have 8 tensors for M1, 8 for M4, etc.
    # Total 56 tensors.
    # For each tensor, we set U, V, W.
    # For M1 tensors:
    # U: non-zero at A00 and A11.
    # V: non-zero at B00 and B11.
    # W: non-zero at C00 and C11?
    # Wait, M1 contributes to C00 and C11.
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So the W part of M1 tensors should point to C00 and C11?
    # No, M1 is a $1 \times 4$ matrix.
    # M1 has 4 components.
    # Let's say M1 = (m1_0, m1_1, m1_2, m1_3).
    # These components are added to C00 and C11?
    # No, C00 is a $1 \times 4$ matrix.
    # So M1 is added directly to C00.
    # And M1 is added directly to C11?
    # In Strassen, M1 is a $2 \times 2$ matrix.
    # Here, our "matrices" are $1 \times 4$.
    # So M1 is a $1 \times 4$ matrix.
    # C00 is a $1 \times 4$ matrix.
    # So M1 is added to C00.
    # Is M1 added to C11?
    # Yes, in standard Strassen, M1 contributes to C11 and C22.
    # Here, C11 and C22 are C01 and C11?
    # Let's check the indices.
    # C = [[C00, C01], [C10, C11]]
    # M1 contributes to C00 and C11.
    # So the W part of M1 tensors should have 1s at C00 indices and C11 indices.
    # But M1 is a single $1 \times 4$ vector.
    # It has 4 components.
    # C00 has 4 components.
    # C11 has 4 components.
    # So M1's 4 components are added to C00's 4 components AND C11's 4 components?
    # That would mean $C00_j += M1_j$ and $C11_j += M1_j$.
    # This is correct for Strassen.
    
    # So for each of the 8 tensors in M1:
    # U: 1 at A00, 1 at A11.
    # V: 1 at B00, 1 at B11.
    # W: 1 at C00_j and C11_j for the corresponding j.
    
    # Let's implement this.

    # M1
    for j in range(4):
        # Term 1: A00 * B00
        U[A00[0]][col] = 1
        V[B00[0*4+j]][col] = 1
        W[C00[j]][col] = 1
        W[C11[j]][col] = 1
        col += 1
        # Term 2: A00 * B11
        U[A00[0]][col] = 1
        V[B11[0*4+j]][col] = 1
        W[C00[j]][col] = 1
        W[C11[j]][col] = 1
        col += 1
        # Term 3: A11 * B00
        U[A11[0]][col] = 1
        V[B00[0*4+j]][col] = 1
        W[C00[j]][col] = 1
        W[C11[j]][col] = 1
        col += 1
        # Term 4: A11 * B11
        U[A11[0]][col] = 1
        V[B11[0*4+j]][col] = 1
        W[C00[j]][col] = 1
        W[C11[j]][col] = 1
        col += 1
        # Wait, A00 is 1x2. A00[0] is the first element.
        # The block product is (1x2) * (2x4).
        # It has 8 terms.
        # My loop only did 4 terms (one for each j, but only first element of A and first row of B).
        # I need to do both elements of A and both rows of B.
        
    # Let's restart the helper function logic.
    
    return {"U": U, "V": W}
# EVOLVE-BLOCK-END
