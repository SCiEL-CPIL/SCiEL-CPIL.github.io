import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Embed 3x5x6 into 4x8x8
    # A (3x5) -> A' (4x8)
    # B (5x6) -> B' (8x8)
    # C (3x6) -> C' (4x8)
    
    # A' = [A_padded, 0] where A_padded is 4x5? No.
    # A is 3x5. Pad to 4x8.
    # A' = [[A, 0], [0, 0]]?
    # A' (4x8).
    # B' (8x8).
    # B' = [[B_padded, 0], [0, 0]]?
    # B is 5x6. Pad to 8x8.
    # B' = [[B, 0], [0, 0]]?
    # B is 5x6.
    # B' = 
    #   rows 0-4: cols 0-5 are B, cols 6-7 are 0
    #   rows 5-7: all 0
    
    # C' = A' B'
    # C' (4x8).
    # C' should contain C (3x6) in the top-left 3x6 block.
    
    # Let's define the mapping.
    # U indices: 4*8 = 32.
    # V indices: 8*8 = 64.
    # W indices: 8*4 = 32.
    
    # We decompose 4x8x8 into four 4x4x4 blocks.
    # A' = [A_L, A_R] (4x4 each)
    # B' = [B_T; B_B] (4x8 each)
    # C' = A_L B_T + A_R B_B
    
    # Block 1: A_L (4x4) * B_T (4x8) -> C_T (4x8)
    # Block 2: A_R (4x4) * B_B (4x8) -> C_B (4x8)
    
    # R(4,4,8) <= 2 * R(4,4,4) = 32.
    # Total rank = 32 + 32 = 64.
    
    rank = 64
    U = [[0] * rank for _ in range(4 * 8)]
    V = [[0] * rank for _ in range(8 * 8)]
    W = [[0] * rank for _ in range(8 * 4)]
    
    # Helper to add a 4x4x4 schoolbook block
    # u_offset: start index in U (4*4=16 entries)
    # v_offset: start index in V (4*4=16 entries)
    # w_offset: start index in W (4*4=16 entries)
    # u_map: list of 16 U indices
    # v_map: list of 16 V indices
    # w_map: list of 16 W indices
    
    def add_schoolbook_block(u_indices, v_indices, w_indices, rank_offset):
        for idx in range(16):
            u = u_indices[idx]
            v = v_indices[idx]
            w = w_indices[idx]
            col = rank_offset + idx
            U[u][col] = 1
            V[v][col] = 1
            W[w][col] = 1

    # Block 1: A_L (rows 0-3, cols 0-3 of A') * B_T (rows 0-3, cols 0-7 of B')
    # A_L is 4x4. B_T is 4x8.
    # We split B_T into two 4x4 blocks: B_TL (cols 0-3), B_TR (cols 4-7).
    # C_TL = A_L * B_TL
    # C_TR = A_L * B_TR
    
    # Sub-block 1.1: A_L * B_TL
    # A_L indices in A':
    # A' is 4x8. A_L is cols 0-3.
    # U indices for A_L:
    # A' row i, col j -> U[i*8 + j]
    # A_L: i in 0-3, j in 0-3.
    u_indices_11 = [i*8 + j for i in range(4) for j in range(4)]
    
    # B_TL indices in B':
    # B' is 8x8. B_T is rows 0-3. B_TL is cols 0-3.
    # V indices for B_TL:
    # B' row i, col j -> V[i*8 + j]
    # B_TL: i in 0-3, j in 0-3.
    v_indices_11 = [i*8 + j for i in range(4) for j in range(4)]
    
    # C_TL indices in C':
    # C' is 4x8. C_T is rows 0-3. C_TL is cols 0-3.
    # W indices for C_TL:
    # C' row i, col j -> W[i*8 + j]
    # C_TL: i in 0-3, j in 0-3.
    w_indices_11 = [i*8 + j for i in range(4) for j in range(4)]
    
    add_schoolbook_block(u_indices_11, v_indices_11, w_indices_11, 0)
    
    # Sub-block 1.2: A_L * B_TR
    # A_L same as above.
    u_indices_12 = u_indices_11
    
    # B_TR: rows 0-3, cols 4-7.
    v_indices_12 = [i*8 + (j+4) for i in range(4) for j in range(4)]
    
    # C_TR: rows 0-3, cols 4-7.
    w_indices_12 = [i*8 + (j+4) for i in range(4) for j in range(4)]
    
    add_schoolbook_block(u_indices_12, v_indices_12, w_indices_12, 16)
    
    # Block 2: A_R * B_B
    # A_R: rows 0-3, cols 4-7 of A'.
    u_indices_21 = [i*8 + (j+4) for i in range(4) for j in range(4)]
    
    # B_B: rows 4-7, cols 0-7 of B'.
    # Split into B_BL (cols 0-3) and B_BR (cols 4-7).
    
    # Sub-block 2.1: A_R * B_BL
    # B_BL: rows 4-7, cols 0-3.
    v_indices_21 = [(i+4)*8 + j for i in range(4) for j in range(4)]
    
    # C_BL: rows 0-3, cols 0-3 of C'?
    # Wait, C' = A_L B_T + A_R B_B.
    # C' is 4x8.
    # A_L B_T gives rows 0-3.
    # A_R B_B gives rows 0-3.
    # So C' rows 0-3 are the sum.
    # C_TL (from 1.1) is cols 0-3.
    # C_BL (from 2.1) is cols 0-3.
    # They add up to C'[0:4, 0:4].
    # So W indices are the same.
    w_indices_21 = w_indices_11
    
    add_schoolbook_block(u_indices_21, v_indices_21, w_indices_21, 32)
    
    # Sub-block 2.2: A_R * B_BR
    # B_BR: rows 4-7, cols 4-7.
    v_indices_22 = [(i+4)*8 + (j+4) for i in range(4) for j in range(4)]
    
    # C_BR: rows 0-3, cols 4-7.
    w_indices_22 = w_indices_12
    
    add_schoolbook_block(u_indices_22, v_indices_22, w_indices_22, 48)
    
    # Now, we need to map the original 3x5x6 problem to this 4x8x8 embedding.
    # The U, V, W matrices currently decompose the 4x8x8 multiplication.
    # We need to restrict them to the 3x5x6 sub-space.
    
    # Original U: 3*5 = 15 entries.
    # Original V: 5*6 = 30 entries.
    # Original W: 6*3 = 18 entries.
    
    # We need to select the relevant rows from the large U, V, W.
    
    # U_orig[i, j] corresponds to U_large[i*8 + j] for i in 0-2, j in 0-4.
    # V_orig[i, j] corresponds to V_large[i*8 + j] for i in 0-4, j in 0-5.
    # W_orig[i, j] corresponds to W_large[i*8 + j] for i in 0-5, j in 0-2.
    # Note: W is p*n = 6*3 = 18.
    # W_large is 8*4 = 32.
    # W_large index: row k (0-7), col i (0-3).
    # W_orig index: row k (0-5), col i (0-2).
    
    U_orig = []
    for i in range(3):
        for j in range(5):
            U_orig.append(U[i*8 + j])
            
    V_orig = []
    for i in range(5):
        for j in range(6):
            V_orig.append(V[i*8 + j])
            
    W_orig = []
    for k in range(6):
        for i in range(3):
            W_orig.append(W[k*8 + i])
            
    return {"U": U_orig, "V": V_orig, "W": W_orig}
# EVOLVE-BLOCK-END
