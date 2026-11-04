import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition for 4x4x4
    # Strassen 2x2 factors
    # u, v, w are 7x4
    u2 = np.array([
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    v2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    w2 = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0]
    ])
    
    # Coefficients for A and B blocks in the 7 Strassen steps
    # Steps 1..7
    # A blocks: 11, 12, 21, 22
    # Indices in 4x4 A:
    # 11: 0,1,2,3
    # 12: 4,5,6,7
    # 21: 8,9,10,11
    # 22: 12,13,14,15
    
    # cA[s][block]
    cA = [
        {11: 1, 22: 1},
        {21: 1, 22: 1},
        {11: 1},
        {22: 1},
        {11: 1, 12: 1},
        {21: 1, 11: -1},
        {12: 1, 22: -1}
    ]
    cB = [
        {11: 1, 22: 1},
        {11: 1},
        {12: 1, 22: -1},
        {21: 1, 11: -1},
        {22: 1},
        {11: 1, 12: 1},
        {21: 1, 22: 1}
    ]
    
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    block_offsets = {
        11: [0, 1, 2, 3],
        12: [4, 5, 6, 7],
        21: [8, 9, 10, 11],
        22: [12, 13, 14, 15]
    }
    
    for s in range(7):
        # For each of the 7 scalar multiplications in the 2x2 Strassen
        for c in range(7):
            col = s * 7 + c
            
            # Construct U vector for this column
            # u2[c] is the linear form on the 2x2 A matrix
            # We need to map this to the 4x4 A vector
            u_val = u2[c]
            for block, coeff in cA[s].items():
                offsets = block_offsets[block]
                for k in range(4):
                    U44[offsets[k], col] += coeff * u_val[k]
            
            # Construct V vector for this column
            v_val = v2[c]
            for block, coeff in cB[s].items():
                offsets = block_offsets[block]
                for k in range(4):
                    V44[offsets[k], col] += coeff * v_val[k]
            
            # Construct W vector for this column
            # w2[c] is the contribution to the 2x2 C matrix
            # The 2x2 C matrix is one of the blocks of the 4x4 C matrix
            # We need to determine which block of C is being computed
            # and map the 2x2 result to the 4x4 C vector
            # The 7 Strassen steps compute linear combinations of C blocks.
            # Actually, the W matrix in the tensor decomposition handles the final assembly.
            # In the block Strassen approach, the "W" part of the 2x2 Strassen
            # gives the contribution to the local 2x2 result.
            # We need to sum these contributions into the global C.
            
            # The local 2x2 result is a matrix M_s.
            # The global C is formed by:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for step s, the local 2x2 matrix M_s is added to the global C
            # with coefficients determined by the Strassen assembly formulas.
            
            # Let's define the assembly coefficients for each step s (0-indexed)
            # s=0 (M1): C11+=1, C22+=1
            # s=1 (M2): C21+=1, C22-=1
            # s=2 (M3): C12+=1, C22+=1
            # s=3 (M4): C11+=1, C21+=1
            # s=4 (M5): C11-=1, C12+=1
            # s=5 (M6): C22+=1
            # s=6 (M7): C11+=1, C12+=1
            
            # The local 2x2 matrix indices (row, col) in the block:
            # 00, 01, 10, 11
            # Map to global C block:
            # Block 11 (top-left): global rows 0-1, cols 0-1
            # Block 12 (top-right): global rows 0-1, cols 2-3
            # Block 21 (bottom-left): global rows 2-3, cols 0-1
            # Block 22 (bottom-right): global rows 2-3, cols 2-3
            
            # w2[c] is a vector [C00, C10, C01, C11] of the local 2x2 matrix.
            # Note: w2 uses column-major flattening?
            # My previous w2 derivation:
            # w2[0] = [1, 0, 0, 1] -> C00=1, C10=0, C01=0, C11=1.
            # This corresponds to local matrix [[1, 0], [0, 1]]?
            # No, C00 is (0,0), C10 is (1,0), C01 is (0,1), C11 is (1,1).
            # So w2[c] = [M00, M10, M01, M11].
            
            w_val = w2[c]
            # w_val[0] is local (0,0)
            # w_val[1] is local (1,0)
            # w_val[2] is local (0,1)
            # w_val[3] is local (1,1)
            
            # Define assembly rules for step s
            # s=0: C11 += M, C22 += M
            # s=1: C21 += M, C22 -= M
            # s=2: C12 += M, C22 += M
            # s=3: C11 += M, C21 += M
            # s=4: C11 -= M, C12 += M
            # s=5: C22 += M
            # s=6: C11 += M, C12 += M
            
            rules = [
                [(11, 1), (22, 1)],
                [(21, 1), (22, -1)],
                [(12, 1), (22, 1)],
                [(11, 1), (21, 1)],
                [(11, -1), (12, 1)],
                [(22, 1)],
                [(11, 1), (12, 1)]
            ]
            
            for block, coeff in rules[s]:
                # Map local 2x2 indices to global 4x4 indices
                # Global C vector index is k*n + i = 4*k + i.
                # Local (r, c) in block (br, bc) maps to global (br*2+r, bc*2+c).
                # Global row i = br*2 + r
                # Global col k = bc*2 + c
                # Global index = 4*k + i = 4*(bc*2+c) + (br*2+r)
                
                # w_val indices:
                # 0: local (0,0) -> r=0, c=0
                # 1: local (1,0) -> r=1, c=0
                # 2: local (0,1) -> r=0, c=1
                # 3: local (1,1) -> r=1, c=1
                
                br, bc = divmod(block, 10) # 11 -> 1,1; 12 -> 1,2; 21 -> 2,1; 22 -> 2,2
                # Wait, 11 means row 1, col 1?
                # My block notation: 11 is top-left (row 0, col 0 in 2x2 block grid).
                # Let's use 0-indexed blocks:
                # 0: 11 (top-left)
                # 1: 12 (top-right)
                # 2: 21 (bottom-left)
                # 3: 22 (bottom-right)
                
                # Re-mapping rules to 0-indexed blocks
                # s=0: 0 += M, 3 += M
                # s=1: 2 += M, 3 -= M
                # s=2: 1 += M, 3 += M
                # s=3: 0 += M, 2 += M
                # s=4: 0 -= M, 1 += M
                # s=5: 3 += M
                # s=6: 0 += M, 1 += M
                
                # I used 11, 12 etc in the rules list above.
                # Let's convert block 11 to 0, 12 to 1, 21 to 2, 22 to 3.
                
                pass # Handled in the loop below

            # Redo the loop with 0-indexed blocks
            block_map = {11: 0, 12: 1, 21: 2, 22: 3}
            for block, coeff in rules[s]:
                b_idx = block_map[block]
                br, bc = divmod(b_idx, 2)
                
                for local_idx in range(4):
                    # local_idx 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)
                    r, c = divmod(local_idx, 2) if local_idx in [0,1] else (local_idx//2, local_idx%2)
                    # Wait, local_idx 0 -> (0,0). 1 -> (1,0). 2 -> (0,1). 3 -> (1,1).
                    # r = local_idx // 2 ? No.
                    # 0: 0//2=0, 0%2=0 -> (0,0)
                    # 1: 1//2=0, 1%2=1 -> (0,1) WRONG.
                    # My w2 vector order: [C00, C10, C01, C11].
                    # 0: (0,0)
                    # 1: (1,0)
                    # 2: (0,1)
                    # 3: (1,1)
                    # So r = local_idx % 2 ? No.
                    # 0: r=0, c=0
                    # 1: r=1, c=0
                    # 2: r=0, c=1
                    # 3: r=1, c=1
                    # r = local_idx % 2
                    # c = local_idx // 2
                    r = local_idx % 2
                    c = local_idx // 2
                    
                    g_i = br * 2 + r
                    g_k = bc * 2 + c
                    g_idx = g_k * n + g_i
                    
                    W44[g_idx, col] += coeff * w_val[local_idx]

    # Construct full matrices
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    U[:, 0:49] = U44
    U[:, 49:98] = U44
    
    V[0:16, 0:49] = V44
    V[16:32, 49:98] = V44
    
    W[0:16, 0:49] = W44
    W[16:32, 49:98] = W44
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
