"""Matrix multiplication 6,6,6 - Strassen-based decomposition"""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # For 6x6 matrix multiplication, use Strassen's algorithm on 2x2 blocks of 3x3 matrices
    # Strassen 2x2: 7 multiplications of 3x3 matrices
    # 3x3 matrix multiplication: rank 19 (known minimal)
    # Total rank: 7 * 19 = 133
    
    # First, define the 3x3 matrix multiplication tensor of rank 19
    # Using the known decomposition for 3x3 matrix multiplication
    # The tensor T_{ijk} for (i,j) in 3x3 output, k in 3x3 input pairs
    # We need U3, V3, W3 for 3x3 matmul with rank 19
    
    # 3x3 matrix multiplication: C = A * B where A is 3x3, B is 3x3
    # U3 has shape (3*3, 19) = (9, 19)
    # V3 has shape (3*3, 19) = (9, 19)  
    # W3 has shape (3*3, 19) = (9, 19) with convention W[k*3+i, r]
    
    # Known rank-19 decomposition for 3x3 matrix multiplication
    # This is based on the work by Landsberg and others
    # Let me use a standard decomposition
    
    # For 3x3 matmul, one known rank-19 decomposition:
    # The 9 output entries C[i][j] for i,j in 0,1,2
    # Each is a sum of products of entries from A and B
    
    # I'll use the explicit rank-19 decomposition
    # Reference: The tensor rank of 3x3 matrix multiplication is 19
    
    # Let me define it using the known structure
    # The 19 terms correspond to specific linear combinations
    
    # Actually, let me use a simpler approach: build the 6x6 decomposition
    # directly using Strassen's algorithm applied to 2x2 blocks of 3x3 matrices
    
    # Strassen 2x2 formulas:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 6x6, split into 2x2 blocks of 3x3:
    # A = [[A11, A12], [A21, A22]] where each is 3x3
    # B = [[B11, B12], [B21, B22]] where each is 3x3
    # C = [[C11, C12], [C21, C22]] where each is 3x3
    
    # Each Mi is a 3x3 matrix product, which has rank 19
    # So we have 7 such products, each contributing 19 to the rank
    # Total rank = 7 * 19 = 133
    
    # Now I need the actual rank-19 decomposition for 3x3 matrix multiplication
    # Let me use the known explicit decomposition
    
    # The 3x3 matrix multiplication tensor T_3: T_3[i,j,k] where
    # i = row of C (0,1,2), j = col of C (0,1,2), and the input indices
    # For the tensor decomposition, we need:
    # U3[a, r] where a = row*3+col of A (0..8), r = 0..18
    # V3[b, r] where b = row*3+col of B (0..8), r = 0..18
    # W3[c, r] where c = row*3+col of C (0..8), r = 0..18
    # with W3 convention: W3[k*n+i, r] where for 3x3, this is W3[col*3+row, r]
    
    # Wait, the convention is W[k*n+i, r] for the output.
    # For 3x3: output C has rows i=0,1,2 and cols j=0,1,2
    # W index: k*n+i where k=col (j), n=3, i=row (i)
    # So W3[j*3+i, r]
    
    # Let me use a known rank-19 decomposition for 3x3 matmul
    # I'll construct it from the bilinear form
    
    # For 3x3: C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor has 9*9 = 81 entries in the bilinear form
    # We need rank-19 decomposition
    
    # Here's a known rank-19 decomposition (from literature):
    # I'll define the 19 terms explicitly
    
    # Let me use the decomposition where:
    # The 19 rank-1 tensors correspond to specific linear combinations
    
    # Actually, let me just build it computationally using the known fact
    # that 3x3 matmul has rank 19, and use a standard decomposition
    
    # For simplicity and correctness, let me use the following approach:
    # I'll define the 3x3 matmul tensor decomposition with rank 19
    # using the explicit formulas from the literature
    
    # Here's a rank-19 decomposition for 3x3 matrix multiplication:
    # (This is one known decomposition)
    
    # Let me define it as a list of (u, v, w) triples where
    # u is a vector in R^9 (for A), v is a vector in R^9 (for B), 
    # w is a vector in R^9 (for C)
    
    # The decomposition: T = sum_{r=0}^{18} u_r ⊗ v_r ⊗ w_r
    
    # I'll use the following known rank-19 decomposition:
    # (Based on the structure of 3x3 matmul)
    
    # For now, let me use a simpler but valid approach:
    # Use the schoolbook for 3x3 (rank 27) which gives total rank 7*27 = 187
    # But that's not optimal. Let me try to get rank 19 for 3x3.
    
    # Actually, let me just use the schoolbook for the 3x3 blocks to ensure correctness,
    # giving rank 7 * 27 = 187, which is still better than 216.
    # Then I can try to improve the 3x3 part.
    
    # For now, use schoolbook 3x3 (rank 27) for correctness
    # This gives total rank = 7 * 27 = 187
    
    # But let me try to get a better 3x3 decomposition.
    # The minimal rank for 3x3 is 19, so let me use that.
    
    # Here's a rank-19 decomposition for 3x3 matrix multiplication:
    # I'll define it explicitly
    
    # The 9 entries of A: A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2], A[2,0], A[2,1], A[2,2]
    # Indexed as 0,1,2,3,4,5,6,7,8
    
    # The 9 entries of B: B[0,0], B[0,1], B[0,2], B[1,0], B[1,1], B[1,2], B[2,0], B[2,1], B[2,2]
    # Indexed as 0,1,2,3,4,5,6,7,8
    
    # The 9 entries of C: C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2], C[2,0], C[2,1], C[2,2]
    # Indexed as 0,1,2,3,4,5,6,7,8
    # W convention: W[k*3+i] where k=col, i=row, so W[j*3+i]
    
    # Rank-19 decomposition (one known decomposition):
    # I'll use the following structure based on the bilinear form
    
    # For correctness, let me first implement with schoolbook 3x3 (rank 27)
    # giving total rank 187, then try to optimize
    
    rank_3x3 = 27  # schoolbook for now
    total_rank = 7 * rank_3x3  # = 187
    
    # Build the 3x3 schoolbook decomposition
    U3 = [[0] * rank_3x3 for _ in range(9)]
    V3 = [[0] * rank_3x3 for _ in range(9)]
    W3 = [[0] * rank_3x3 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U3[i * 3 + k][col] = 1  # A[i,k]
                V3[k * 3 + j][col] = 1  # B[k,j]
                W3[j * 3 + i][col] = 1  # C[i,j] with W[j*3+i]
                col += 1
    
    # Now build the 6x6 decomposition using Strassen on 2x2 blocks of 3x3
    # The 6x6 matrices are split into 2x2 blocks of 3x3 matrices
    # A blocks: A11 (rows 0-2, cols 0-2), A12 (rows 0-2, cols 3-5), 
    #           A21 (rows 3-5, cols 0-2), A22 (rows 3-5, cols 3-5)
    # B blocks: B11 (rows 0-2, cols 0-2), B12 (rows 0-2, cols 3-5),
    #           B21 (rows 3-5, cols 0-2), B22 (rows 3-5, cols 3-5)
    # C blocks: C11 (rows 0-2, cols 0-2), C12 (rows 0-2, cols 3-5),
    #           C21 (rows 3-5, cols 0-2), C22 (rows 3-5, cols 3-5)
    
    # Strassen formulas (for 2x2 block matrices):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 3x3 matrix product, decomposed with rank 27
    # So for each Mi, we have 27 rank-1 terms
    
    # The full 6x6 decomposition:
    # U has shape (36, 187), V has shape (36, 187), W has shape (36, 187)
    # 36 = 6*6
    
    # For each Strassen term Mi (i=1..7), and for each of the 27 rank-1 terms in the 3x3 decomposition:
    # We need to map the 3x3 block entries to the 6x6 entries
    
    # Let me define the mapping for each block:
    # For a 3x3 block at position (block_row, block_col) in the 2x2 block structure:
    # The 3x3 block occupies rows [block_row*3, block_row*3+2] and cols [block_col*3, block_col*3+2]
    
    # For A (input 1): A[i,j] where i=row, j=col, index = i*6+j
    # Block A11: rows 0-2, cols 0-2 -> A indices: 0,1,2,6,7,8,12,13,14
    # Block A12: rows 0-2, cols 3-5 -> A indices: 3,4,5,9,10,11,15,16,17
    # Block A21: rows 3-5, cols 0-2 -> A indices: 18,19,20,24,25,26,30,31,32
    # Block A22: rows 3-5, cols 3-5 -> A indices: 21,22,23,27,28,29,33,34,35
    
    # For B (input 2): B[i,j] where i=row, j=col, index = i*6+j
    # Block B11: rows 0-2, cols 0-2 -> B indices: 0,1,2,6,7,8,12,13,14
    # Block B12: rows 0-2, cols 3-5 -> B indices: 3,4,5,9,10,11,15,16,17
    # Block B21: rows 3-5, cols 0-2 -> B indices: 18,19,20,24,25,26,30,31,32
    # Block B22: rows 3-5, cols 3-5 -> B indices: 21,22,23,27,28,29,33,34,35
    
    # For C (output): C[i,j] where i=row, j=col
    # W convention: W[k*6+i, r] where k=col, i=row
    # So W index for C[i,j] is j*6+i
    # Block C11: rows 0-2, cols 0-2 -> C[i,j] for i,j in 0,1,2 -> W indices: j*6+i
    #   C[0,0]->W[0], C[0,1]->W[6], C[0,2]->W[12]
    #   C[1,0]->W[1], C[1,1]->W[7], C[1,2]->W[13]
    #   C[2,0]->W[2], C[2,1]->W[8], C[2,2]->W[14]
    # Block C12: rows 0-2, cols 3-5 -> C[i,j] for i in 0,1,2, j in 3,4,5
    #   C[0,3]->W[18], C[0,4]->W[24], C[0,5]->W[30]
    #   C[1,3]->W[19], C[1,4]->W[25], C[1,5]->W[31]
    #   C[2,3]->W[20], C[2,4]->W[26], C[2,5]->W[32]
    # Block C21: rows 3-5, cols 0-2 -> C[i,j] for i in 3,4,5, j in 0,1,2
    #   C[3,0]->W[3], C[3,1]->W[9], C[3,2]->W[15]
    #   C[4,0]->W[4], C[4,1]->W[10], C[4,2]->W[16]
    #   C[5,0]->W[5], C[5,1]->W[11], C[5,2]->W[17]
    # Block C22: rows 3-5, cols 3-5 -> C[i,j] for i in 3,4,5, j in 3,4,5
    #   C[3,3]->W[21], C[3,4]->W[27], C[3,5]->W[33]
    #   C[4,3]->W[22], C[4,4]->W[28], C[4,5]->W[34]
    #   C[5,3]->W[23], C[5,4]->W[29], C[5,5]->W[35]
    
    # Helper function to get the 6x6 index for a 3x3 block entry
    def a_block_index(block_row, block_col, local_i, local_j):
        """Get 6x6 index for A entry at block (block_row, block_col), local position (local_i, local_j)"""
        global_i = block_row * 3 + local_i
        global_j = block_col * 3 + local_j
        return global_i * 6 + global_j
    
    def b_block_index(block_row, block_col, local_i, local_j):
        """Get 6x6 index for B entry at block (block_row, block_col), local position (local_i, local_j)"""
        global_i = block_row * 3 + local_i
        global_j = block_col * 3 + local_j
        return global_i * 6 + global_j
    
    def c_block_index(block_row, block_col, local_i, local_j):
        """Get W index for C entry at block (block_row, block_col), local position (local_i, local_j)
        W convention: W[col*6+row]
        """
        global_i = block_row * 3 + local_i
        global_j = block_col * 3 + local_j
        return global_j * 6 + global_i
    
    # Initialize U, V, W for 6x6
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Strassen terms: for each term, define which A blocks and B blocks are used, and the coefficients
    # Each term is a product of a linear combination of A blocks and a linear combination of B blocks
    # Then the result is a 3x3 matrix that goes into specific C blocks
    
    # Define the 7 Strassen terms:
    # term_idx: (A_blocks_with_coeffs, B_blocks_with_coeffs, C_blocks_with_coeffs)
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    # C_blocks: list of (block_row, block_col, coeff)
    
    strassen_terms = [
        # M1 = (A11 + A22) * (B11 + B22)
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'C': [(0,0,1), (1,1,1)]  # C11 += M1, C22 += M1
        },
        # M2 = (A21 + A22) * B11
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'C': [(1,0,1), (1,1,-1)]  # C21 += M2, C22 -= M2
        },
        # M3 = A11 * (B12 - B22)
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'C': [(0,1,1), (1,1,1)]  # C12 += M3, C22 += M3
        },
        # M4 = A22 * (B21 - B11)
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'C': [(0,0,1), (1,0,1)]  # C11 += M4, C21 += M4
        },
        # M5 = (A11 + A12) * B22
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'C': [(0,0,-1), (0,1,1)]  # C11 -= M5, C12 += M5
        },
        # M6 = (A21 - A11) * (B11 + B12)
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'C': [(1,1,1)]  # C22 += M6
        },
        # M7 = (A12 - A22) * (B21 + B22)
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'C': [(0,0,1)]  # C11 += M7
        },
    ]
    
    # For each Strassen term and each rank-1 term in the 3x3 decomposition,
    # add the contributions to U, V, W
    
    for term_idx, term in enumerate(strassen_terms):
        for r3 in range(rank_3x3):
            r = term_idx * rank_3x3 + r3  # global rank index
            
            # For U: sum over A blocks
            for (ar, ac, coeff_a) in term['A']:
                # For each entry in the 3x3 block
                for li in range(3):
                    for lj in range(3):
                        # The 3x3 decomposition entry U3[li*3+lj, r3]
                        u3_val = U3[li * 3 + lj][r3]
                        if u3_val != 0:
                            a_idx = a_block_index(ar, ac, li, lj)
                            U[a_idx][r] += coeff_a * u3_val
            
            # For V: sum over B blocks
            for (br, bc, coeff_b) in term['B']:
                for li in range(3):
                    for lj in range(3):
                        v3_val = V3[li * 3 + lj][r3]
                        if v3_val != 0:
                            b_idx = b_block_index(br, bc, li, lj)
                            V[b_idx][r] += coeff_b * v3_val
            
            # For W: sum over C blocks
            for (cr, cc, coeff_c) in term['C']:
                for li in range(3):
                    for lj in range(3):
                        w3_val = W3[li * 3 + lj][r3]
                        if w3_val != 0:
                            w_idx = c_block_index(cr, cc, li, lj)
                            W[w_idx][r] += coeff_c * w3_val
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
