import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # The rank of 4x4x4 is 49.
    # We construct the 49 terms for the first 4 columns of B (and C).
    # Then we duplicate for the next 4 columns.
    
    # Strassen 2x2x2 coefficients
    # U22, V22, W22 are 4x7
    # Rows are local indices 0..3 (00, 01, 10, 11)
    # Cols are term indices 0..6
    
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11)
    U22[0,0] = 1; U22[3,0] = 1
    V22[0,0] = 1; V22[3,0] = 1
    W22[0,0] = 1; W22[3,0] = 1
    
    # Term 1: (A10+A11)B00
    U22[2,1] = 1; U22[3,1] = 1
    V22[0,1] = 1
    W22[2,1] = 1; W22[3,1] = -1
    
    # Term 2: A00(B01-B11)
    U22[0,2] = 1
    V22[1,2] = 1; V22[3,2] = -1
    W22[1,2] = 1; W22[3,2] = 1
    
    # Term 3: A11(B10-B00)
    U22[3,3] = 1
    V22[2,3] = 1; V22[0,3] = -1
    W22[0,3] = 1; W22[2,3] = 1
    
    # Term 4: (A00+A01)B11
    U22[0,4] = 1; U22[1,4] = 1
    V22[3,4] = 1
    W22[0,4] = -1; W22[1,4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U22[2,5] = 1; U22[0,5] = -1
    V22[0,5] = 1; V22[1,5] = 1
    W22[3,5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    U22[1,6] = 1; U22[3,6] = -1
    V22[2,6] = 1; V22[3,6] = 1
    W22[0,6] = 1

    # Now build 4x4x4 decomposition (49 terms)
    # U44: 16x49, V44: 16x49, W44: 16x49
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    # Map local 2x2 block indices to global 4x4 indices
    # Block 0 (00): rows 0,1 cols 0,1 -> indices 0,1,4,5
    # Block 1 (01): rows 0,1 cols 2,3 -> indices 2,3,6,7
    # Block 2 (10): rows 2,3 cols 0,1 -> indices 8,9,12,13
    # Block 3 (11): rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # Global index for local index (r, c) in block (br, bc) is:
    # (br*2 + r)*4 + (bc*2 + c)
    
    def get_global_indices(block_r, block_c):
        indices = []
        for r in range(2):
            for c in range(2):
                idx = (block_r * 2 + r) * 4 + (block_c * 2 + c)
                indices.append(idx)
        return indices

    # Precompute global indices for each block
    block_indices = {
        0: get_global_indices(0, 0), # 00
        1: get_global_indices(0, 1), # 01
        2: get_global_indices(1, 0), # 10
        3: get_global_indices(1, 1)  # 11
    }

    # Define the 7 outer terms
    # Each term is a tuple of (a_blocks, b_blocks, c_blocks)
    # a_blocks: list of (block_id, weight)
    terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]

    for t_idx in range(7):
        a_parts, b_parts, c_parts = terms[t_idx]
        for s_idx in range(7):
            col = t_idx * 7 + s_idx
            
            # Fill U
            for la in range(4):
                coeff = U22[la, s_idx]
                if coeff != 0:
                    # la is local index in 2x2 block
                    # la // 2 is block_row, la % 2 is block_col
                    # But we need to map to global indices of the blocks in a_parts
                    # Wait, U22[la, s_idx] corresponds to the local A index la.
                    # The local A index la belongs to a specific block?
                    # No, U22 is for the 2x2 matrix of blocks.
                    # So la=0 is block 0, la=1 is block 1, etc.
                    # So if U22[0, s_idx] != 0, it affects Block 0.
                    # If U22[1, s_idx] != 0, it affects Block 1.
                    
                    block_id = la
                    weight_in_outer = 0
                    # Find weight of this block in a_parts
                    for b_id, w in a_parts:
                        if b_id == block_id:
                            weight_in_outer = w
                            break
                    
                    if weight_in_outer != 0:
                        # Add to global indices of this block
                        for g_idx in block_indices[block_id]:
                            U44[g_idx, col] += weight_in_outer * coeff

            # Fill V
            for lb in range(4):
                coeff = V22[lb, s_idx]
                if coeff != 0:
                    block_id = lb
                    weight_in_outer = 0
                    for b_id, w in b_parts:
                        if b_id == block_id:
                            weight_in_outer = w
                            break
                    if weight_in_outer != 0:
                        for g_idx in block_indices[block_id]:
                            V44[g_idx, col] += weight_in_outer * coeff

            # Fill W
            for lc in range(4):
                coeff = W22[lc, s_idx]
                if coeff != 0:
                    block_id = lc
                    weight_in_outer = 0
                    for c_id, w in c_parts:
                        if c_id == block_id:
                            weight_in_outer = w
                            break
                    if weight_in_outer != 0:
                        for g_idx in block_indices[block_id]:
                            W44[g_idx, col] += weight_in_outer * coeff

    # Now construct 4x4x8
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First half (cols 0-3 of B)
    U[:, 0:49] = U44
    # V indices for first half: j*8 + k (k=0..3)
    # V44 indices: j*4 + k
    for v_idx_44 in range(16):
        j = v_idx_44 // 4
        k = v_idx_44 % 4
        v_idx_global = j * 8 + k
        V[v_idx_global, 0:49] = V44[v_idx_44]
        
    # W indices for first half: k*4 + i (k=0..3)
    # W44 indices: k*4 + i
    W[0:16, 0:49] = W44

    # Second half (cols 4-7 of B)
    U[:, 49:98] = U44
    # V indices for second half: j*8 + (k+4)
    for v_idx_44 in range(16):
        j = v_idx_44 // 4
        k = v_idx_44 % 4
        v_idx_global = j * 8 + (k + 4)
        V[v_idx_global, 49:98] = V44[v_idx_44]
        
    # W indices for second half: (k+4)*4 + i
    for w_idx_44 in range(16):
        k = w_idx_44 // 4
        i = w_idx_44 % 4
        w_idx_global = (k + 4) * 4 + i
        W[w_idx_global, 49:98] = W44[w_idx_44]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
