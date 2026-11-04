"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns a rank-98 exact tensor decomposition for 4x4 by 4x8 matrix multiplication.
    
    We split the 4x8 matrix into two 4x4 blocks and use the rank-49 Strassen-based
    decomposition for each 4x4 multiplication.
    
    U: shape (16, 98) - linear forms on A (4x4)
    V: shape (32, 98) - linear forms on B (4x8), split into two 4x4 blocks
    W: shape (32, 98) - linear forms on C (4x8), split into two 4x4 blocks
    
    The trace convention for W is W[k*n+i, r] where n=4, so W has shape (p*n, rank) = (32, 98).
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2 decomposition: 7 bilinear forms
    # Each form is (u, v) where u and v are 4-dim vectors (flattened 2x2 matrices)
    # u corresponds to linear form on left 2x2 matrix
    # v corresponds to linear form on right 2x2 matrix
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1]), # P1 = (a00+a11)(b00+b11)
        ([0, 0, 1, 1], [1, 0, 0, 0]), # P2 = (a10+a11)b00
        ([1, 0, 0, 0], [0, 1, 0, -1]), # P3 = a00(b01-b11)
        ([0, 0, 0, 1], [0, 0, 1, -1]), # P4 = a11(b10-b00)
        ([1, 1, 0, 0], [0, 0, 0, 1]), # P5 = (a00+a01)b11
        ([0, 0, 1, -1], [1, 1, 0, 0]), # P6 = (a10-a00)(b00+b01)
        ([0, 1, 0, -1], [0, 0, 1, 1]), # P7 = (a01-a11)(b10+b11)
    ]
    
    # Reconstruction: c_ij = sum_k coeff[k] * Pk
    # c00: P1 - P4 + P6 -> [1, 0, 0, -1, 0, 1, 0]
    # c01: P3 + P5      -> [0, 0, 1, 0, 1, 0, 0]
    # c10: P2 + P7      -> [0, 1, 0, 0, 0, 0, 1]
    # c11: P1 - P3 + P4 -> [1, 0, -1, 1, 0, 0, 0]
    recon_2x2 = [
        [1, 0, 0, -1, 0, 1, 0],  # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 0, 0, 0, 1],  # c10
        [1, 0, -1, 1, 0, 0, 0]  # c11
    ]
    
    # Block-level Strassen for 4x4 multiplication:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2.
    # 
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # Reconstruction:
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M3 + M4
    
    # For each block product Mi, we need to express the left and right operands
    # as linear forms on the 16 entries of A and B respectively.
    #
    # The left operand of Mi is a 2x2 matrix. Each entry is a linear combination
    # of entries from specific A blocks.
    #
    # Let's define for each Mi:
    #   L_i: a 2x2 matrix where L_i[k,l] is a 4-dim vector (linear form on 4 entries of a block)
    #   But actually, L_i[k,l] may involve multiple blocks.
    #
    # Better approach: For each Mi, define the left 2x2 matrix as a sum of scaled blocks.
    # Then apply the 2x2 Strassen u-vector to get a 16-dim linear form on A.
    #
    # Let's define the block combinations for L_i and R_i:
    # Each is a 2x2 matrix of block references with coefficients.
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # For B: 0=B00, 1=B01, 2=B10, 3=B11
    
    # L_i definitions: 2x2 matrix where each entry is a list of (coeff, block_idx)
    L_defs = [
        # M1: L = A00 + A11
        [[[(1, 0), (1, 3)], [(1, 0), (1, 3)]],
         [[(1, 0), (1, 3)], [(1, 0), (1, 3)]]],
        # M2: L = A10 + A11
        [[[(1, 2), (1, 3)], [(1, 2), (1, 3)]],
         [[(1, 2), (1, 3)], [(1, 2), (1, 3)]]],
        # M3: L = A00
        [[[(1, 0)], [(1, 0)]],
         [[(1, 0)], [(1, 0)]]],
        # M4: L = A11
        [[[(1, 3)], [(1, 3)]],
         [[(1, 3)], [(1, 3)]]],
        # M5: L = A00 + A01
        [[[(1, 0), (1, 1)], [(1, 0), (1, 1)]],
         [[(1, 0), (1, 1)], [(1, 0), (1, 1)]]],
        # M6: L = A10 - A00
        [[[(1, 2), (-1, 0)], [(1, 2), (-1, 0)]],
         [[(1, 2), (-1, 0)], [(1, 2), (-1, 0)]]],
        # M7: L = A01 - A11
        [[[(1, 1), (-1, 3)], [(1, 1), (-1, 3)]],
         [[(1, 1), (-1, 3)], [(1, 1), (-1, 3)]]],
    ]
    
    R_defs = [
        # M1: R = B00 + B11
        [[[(1, 0), (1, 3)], [(1, 0), (1, 3)]],
         [[(1, 0), (1, 3)], [(1, 0), (1, 3)]]],
        # M2: R = B00
        [[[(1, 0)], [(1, 0)]],
         [[(1, 0)], [(1, 0)]]],
        # M3: R = B01 - B11
        [[[(1, 1), (-1, 3)], [(1, 1), (-1, 3)]],
         [[(1, 1), (-1, 3)], [(1, 1), (-1, 3)]]],
        # M4: R = B10 - B00
        [[[(1, 2), (-1, 0)], [(1, 2), (-1, 0)]],
         [[(1, 2), (-1, 0)], [(1, 2), (-1, 0)]]],
        # M5: R = B11
        [[[(1, 3)], [(1, 3)]],
         [[(1, 3)], [(1, 3)]]],
        # M6: R = B00 + B01
        [[[(1, 0), (1, 1)], [(1, 0), (1, 1)]],
         [[(1, 0), (1, 1)], [(1, 0), (1, 1)]]],
        # M7: R = B10 + B11
        [[[(1, 2), (1, 3)], [(1, 2), (1, 3)]],
         [[(1, 2), (1, 3)], [(1, 2), (1, 3)]]],
    ]
    
    # Block reconstruction: C_ij = sum_k coeff[k] * Mk
    # C00 = M1 - M4 + M6 -> {0: 1, 3: -1, 5: 1}
    # C01 = M3 + M5      -> {2: 1, 4: 1}
    # C10 = M2 + M7      -> {1: 1, 6: 1}
    # C11 = M1 - M3 + M4 -> {0: 1, 2: -1, 3: 1}
    block_recon = [
        {0: 1, 3: -1, 5: 1},  # C00
        {2: 1, 4: 1},          # C01
        {1: 1, 6: 1},          # C10
        {0: 1, 2: -1, 3: 1}    # C11
    ]
    
    # Now generate the 49 scalar products for one 4x4 by 4x4 multiplication.
    # Each product is a triple (u_vec, v_vec, w_vec) where:
    #   u_vec: 16-dim linear form on A
    #   v_vec: 16-dim linear form on B
    #   w_vec: 16-dim linear form on C
    
    products_4x4 = []
    
    for block_idx in range(7):  # 7 block products
        L = L_defs[block_idx]
        R = R_defs[block_idx]
        
        for s_idx in range(7):  # 7 Strassen 2x2 products
            u_2x2 = strassen_2x2[s_idx][0]
            v_2x2 = strassen_2x2[s_idx][1]
            
            # Compute u_vec: 16-dim linear form on A
            u_vec = [0.0] * 16
            for k in range(2):
                for l in range(2):
                    idx = k * 2 + l
                    coeff_2x2 = u_2x2[idx]
                    if coeff_2x2 != 0:
                        # L[k][l] is a list of (coeff, block_idx)
                        for coeff_block, blk in L[k][l]:
                            total_coeff = coeff_2x2 * coeff_block
                            # Block blk has 4 entries: blk*4, blk*4+1, blk*4+2, blk*4+3
                            for entry in range(4):
                                u_vec[blk * 4 + entry] += total_coeff
            
            # Compute v_vec: 16-dim linear form on B
            v_vec = [0.0] * 16
            for k in range(2):
                for l in range(2):
                    idx = k * 2 + l
                    coeff_2x2 = v_2x2[idx]
                    if coeff_2x2 != 0:
                        # R[k][l] is a list of (coeff, block_idx)
                        for coeff_block, blk in R[k][l]:
                            total_coeff = coeff_2x2 * coeff_block
                            for entry in range(4):
                                v_vec[blk * 4 + entry] += total_coeff
            
            # Compute w_vec: 16-dim linear form on C
            # This product contributes to C blocks according to block_recon.
            # For each C block (bi, bj), if block_idx is in block_recon[bi*2+bj],
            # then this product contributes with coefficient block_recon[bi*2+bj][block_idx]
            # times the 2x2 reconstruction coefficient for entry (k,l) of the 2x2 block.
            w_vec = [0.0] * 16
            for bi in range(2):
                for bj in range(2):
                    c_block_idx = bi * 2 + bj
                    if block_idx in block_recon[c_block_idx]:
                        coeff_block = block_recon[c_block_idx][block_idx]
                        # For each entry (k,l) in the 2x2 block:
                        for k in range(2):
                            for l in range(2):
                                s_idx_local = k * 2 + l
                                coeff_2x2 = recon_2x2[s_idx_local][s_idx]
                                if coeff_2x2 != 0:
                                    total_coeff = coeff_block * coeff_2x2
                                    # C entry (bi*2+k, bj*2+l)
                                    c_i = bi * 2 + k
                                    c_j = bj * 2 + l
                                    w_idx = c_i * 4 + c_j
                                    w_vec[w_idx] += total_coeff
            
            products_4x4.append((u_vec, v_vec, w_vec))
    
    # Now we have 49 products for 4x4 by 4x4.
    # For 4x4 by 4x8, we split B into two 4x4 blocks: B_left (cols 0-3) and B_right (cols 4-7).
    # Similarly, C is split into C_left (cols 0-3) and C_right (cols 4-7).
    #
    # We need 98 products total: 49 for C_left and 49 for C_right.
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))  # 4x8 flattened: 32 entries
    W = np.zeros((32, rank))  # 4x8 flattened: 32 entries
    
    # For the left half (cols 0-3 of C):
    # Use the 49 products, with v_vec applied to B_left (first 16 entries of B)
    # and w_vec applied to C_left (first 16 entries of C)
    for i in range(49):
        u_vec, v_vec, w_vec = products_4x4[i]
        col = i
        U[:, col] = u_vec
        V[0:16, col] = v_vec  # B_left
        V[16:32, col] = 0     # B_right
        W[0:16, col] = w_vec  # C_left
        W[16:32, col] = 0     # C_right
    
    # For the right half (cols 4-7 of C):
    # Use the same 49 products, but with v_vec applied to B_right (last 16 entries of B)
    # and w_vec applied to C_right (last 16 entries of C)
    for i in range(49):
        u_vec, v_vec, w_vec = products_4x4[i]
        col = 49 + i
        U[:, col] = u_vec
        V[0:16, col] = 0      # B_left
        V[16:32, col] = v_vec # B_right
        W[0:16, col] = 0      # C_left
        W[16:32, col] = w_vec # C_right
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
