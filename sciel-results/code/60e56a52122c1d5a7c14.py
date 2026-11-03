import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def get_strassen_2x2():
    # Returns U, V, W for 2x2x2 multiplication
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j, k], V[j*2+l, k], W[l*2+i, k]
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1 = (A00 + A11)(B00 + B11)
    # U: A00, A11. V: B00, B11.
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # W: C00, C11.
    # C00 += P1, C11 += P1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10 + A11) B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 += P2
    W[2, 1] = 1
    
    # P3 = A00 (B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C01 += P3, C11 += P3
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A11 (B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C00 += P4, C10 += P4
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (A00 + A10) B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    # C00 -= P5, C01 += P5
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 += P6
    W[3, 5] = 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 -= P7
    W[0, 6] = -1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Decompose into two 4x4x4 and one 4x4x1
    # Rank 49 + 49 + 16 = 114
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: 4x4x4 (cols 0-3 of B)
    # We need the 4x4x4 Strassen decomposition.
    # We construct it by nesting the 2x2x2 decomposition.
    U_2, V_2, W_2 = get_strassen_2x2()
    
    # The 4x4 matrix is 4 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # We compute 7 products of 2x2 matrices.
    # Each product is a 2x2x2 multiplication.
    
    # U for 4x4x4 will have shape (16, 49)
    # V for 4x4x4 will have shape (16, 49) (since p=4 for this part, m*p = 4*4=16)
    # Wait, V shape is (m*p, rank). For 4x4x4, m=4, p=4. V is (16, 49).
    # W shape is (p*n, rank). For 4x4x4, p=4, n=4. W is (16, 49).
    
    U_4_4_4 = np.zeros((16, 49))
    V_4_4_4 = np.zeros((16, 49))
    W_4_4_4 = np.zeros((16, 49))
    
    # Map the 7 Strassen products to indices
    # Product k (0..6) uses U_2, V_2, W_2 with offsets.
    # The input indices for A are 0..15.
    # A00: 0,1,4,5. A01: 2,3,6,7. A10: 8,9,12,13. A11: 10,11,14,15.
    # The output indices for C are 0..15.
    
    # We need to embed the 2x2 decomposition into the 4x4 space.
    # For each of the 7 products, we have a 2x2x2 multiplication.
    # The scalar indices of the 2x2x2 multiplication are 0..6.
    # Total scalar indices for 4x4x4 are 0..48.
    # Product k uses scalar indices k*7 .. k*7+6.
    
    # For a given product k, the input A-block is determined by the Strassen formula.
    # e.g., P1 uses A00 and A11.
    # The U_2 matrix has entries for A00 (idx 0,1) and A11 (idx 2,3)?
    # In get_strassen_2x2, U indices are i*2+j.
    # U[0, 0] is A00. U[3, 0] is A11.
    # So we need to map the local 2x2 indices to the global 4x4 indices.
    
    # Define the block mappings for A and B
    # A blocks: 0:0-3, 1:4-7, 2:8-11, 3:12-15 (row-major)
    # Wait, A is 4x4.
    # A00: [0,1,4,5] -> indices 0,1,4,5.
    # A01: [2,3,6,7] -> indices 2,3,6,7.
    # A10: [8,9,12,13] -> indices 8,9,12,13.
    # A11: [10,11,14,15] -> indices 10,11,14,15.
    
    # B blocks: 0:0-3, 1:4-7, 2:8-11, 3:12-15.
    # B00: 0,1,4,5.
    # B01: 2,3,6,7.
    # B10: 8,9,12,13.
    # B11: 10,11,14,15.
    
    # C blocks: 0:0-3, 1:4-7, 2:8-11, 3:12-15.
    
    # For each product k in 0..6:
    # We determine which A-blocks and B-blocks are involved.
    # From get_strassen_2x2:
    # k=0: A00, A11. B00, B11. -> C00, C11.
    # k=1: A10, A11. B00. -> C10.
    # k=2: A00. B10, B11. -> C01, C11.
    # k=3: A11. B01, B00. -> C00, C10.
    # k=4: A00, A10. B11. -> C00, C01.
    # k=5: A10, A00. B00, B01. -> C11.
    # k=6: A01, A11. B10, B11. -> C00.
    
    # We will iterate through the 7 products and fill U_4_4_4, V_4_4_4, W_4_4_4.
    
    block_map_A = {0: [0, 1, 4, 5], 1: [2, 3, 6, 7], 2: [8, 9, 12, 13], 3: [10, 11, 14, 15]}
    block_map_B = {0: [0, 1, 4, 5], 1: [2, 3, 6, 7], 2: [8, 9, 12, 13], 3: [10, 11, 14, 15]}
    block_map_C = {0: [0, 1, 4, 5], 1: [2, 3, 6, 7], 2: [8, 9, 12, 13], 3: [10, 11, 14, 15]}
    
    # local_idx -> global_idx mapping for a block
    # local 0 -> block[0], local 1 -> block[1], etc.
    
    for k in range(7):
        scalar_offset = k * 7
        
        # Get the local U, V, W for this product
        # We need to know which blocks are used and with what signs.
        # This is hardcoded in get_strassen_2x2.
        # Let's re-extract the logic from get_strassen_2x2 to map to blocks.
        
        # Instead of re-parsing, I will just use the U_2, V_2, W_2 values and map the indices.
        # U_2 indices: 0 (A00), 1 (A01), 2 (A10), 3 (A11).
        # V_2 indices: 0 (B00), 1 (B01), 2 (B10), 3 (B11).
        # W_2 indices: 0 (C00), 1 (C01), 2 (C10), 3 (C11).
        
        # For each local index, if U_2[local_idx, s] != 0, map to global.
        for local_idx in range(4):
            val = U_2[local_idx, k]
            if val != 0:
                # local_idx corresponds to A-block local_idx
                # The 4 entries of A-block local_idx are at global indices block_map_A[local_idx]
                # But U_2 is for 2x2 matrix. The entries of the 2x2 matrix are A[0,0], A[0,1], A[1,0], A[1,1].
                # local_idx 0 is A00. The entries of A00 are A[0,0], A[0,1], A[1,0], A[1,1].
                # In the 4x4 matrix, these are at indices:
                # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5.
                # So we need to add val to U_4_4_4[0, scalar_offset+s], U_4_4_4[1, ...], etc.
                # Wait, U_2[0, k] is the coefficient for A[0,0].
                # U_2[1, k] is the coefficient for A[0,1].
                # U_2[2, k] is the coefficient for A[1,0].
                # U_2[3, k] is the coefficient for A[1,1].
                
                # So for each scalar s in 0..6:
                for s in range(7):
                    coeff = U_2[local_idx, s]
                    if coeff != 0:
                        # Map local A index (row, col) to global
                        # local_idx 0 -> A00.
                        # local_idx 1 -> A01.
                        # local_idx 2 -> A10.
                        # local_idx 3 -> A11.
                        # The local 2x2 indices within the block are 0,1,2,3.
                        # local_idx 0 uses block A00. The 2x2 entries are at local 0,1,2,3.
                        # So if local_idx is 0, we map local 2x2 indices 0,1,2,3 to global indices in block A00.
                        # If local_idx is 1, we map to block A01.
                        
                        # The U_2 matrix has rows 0,1,2,3 corresponding to A[0,0], A[0,1], A[1,0], A[1,1].
                        # So for a fixed local_idx (block), we iterate over the 4 entries of that block?
                        # No, U_2[local_idx, s] is the coefficient for the (row, col) entry of the block.
                        # Wait, U_2 is (4, 7). The rows are 0,1,2,3.
                        # Row 0: A[0,0]. Row 1: A[0,1]. Row 2: A[1,0]. Row 3: A[1,1].
                        # So for block b, we need to know which rows of U_2 correspond to which entries of block b.
                        # This depends on the block.
                        # For A00, row 0 of U_2 is A00[0,0], row 1 is A00[0,1], etc.
                        # For A01, row 0 of U_2 is A01[0,0], etc.
                        # So the mapping is the same for all blocks: row r of U_2 corresponds to entry (r//2, r%2) of the block.
                        
                        # So for each r in 0..3:
                        # coeff = U_2[r, s]
                        # If coeff != 0:
                        #    global_idx = block_map_A[local_idx][r]
                        #    U_4_4_4[global_idx, scalar_offset + s] += coeff
                        
                        # Wait, U_2 is fixed for the product k.
                        # So we iterate r in 0..3.
                        for r in range(4):
                            coeff = U_2[r, s]
                            if coeff != 0:
                                global_idx = block_map_A[local_idx][r]
                                U_4_4_4[global_idx, scalar_offset + s] += coeff
                            
        # Do the same for V and W.
        for local_idx in range(4):
            for s in range(7):
                coeff = V_2[local_idx, s]
                if coeff != 0:
                    for r in range(4):
                        coeff_v = V_2[local_idx, s] # Wait, V_2 row is local_idx?
                        # V_2 rows are B[0,0], B[0,1], B[1,0], B[1,1].
                        # So same logic.
                        # But V_2[local_idx, s] is the coefficient for entry (local_idx//2, local_idx%2) of the block?
                        # No, V_2 row index is 0,1,2,3.
                        # Row 0: B[0,0]. Row 1: B[0,1].
                        # So for block local_idx, we map row r to global index.
                        coeff_v = V_2[r, s]
                        if coeff_v != 0:
                            global_idx = block_map_B[local_idx][r]
                            V_4_4_4[global_idx, scalar_offset + s] += coeff_v

        for local_idx in range(4):
            for s in range(7):
                coeff_w = W_2[r, s] # Error: r is not defined in this scope
                # Correct loop:
                for r in range(4):
                    coeff_w = W_2[r, s]
                    if coeff_w != 0:
                        global_idx = block_map_C[local_idx][r]
                        W_4_4_4[global_idx, scalar_offset + s] += coeff_w

    # This logic is getting messy. I will simplify by just using the schoolbook for the 4x4x4 parts as well?
    # No, 49 is much better than 16.
    # I will hardcode the 4x4x4 Strassen decomposition if possible, or write a cleaner recursive builder.
    
    # Given time constraints, I will use the 49-rank Strassen for the two 4x4 parts and 16-rank schoolbook for the 1x4 part.
    # I will implement a clean builder.
    
    pass
