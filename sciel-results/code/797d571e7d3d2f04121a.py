import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _add_trivial_block(U, V, W, start_rank, i_range, j_range, k_range):
    """
    Adds a trivial decomposition for a block defined by i_range, j_range, k_range.
    Returns the new rank offset.
    """
    rank_offset = start_rank
    for i in i_range:
        for j in j_range:
            for k in k_range:
                U[i * 5 + j, rank_offset] = 1.0
                V[j * 5 + k, rank_offset] = 1.0
                W[k * 5 + i, rank_offset] = 1.0
                rank_offset += 1
    return rank_offset

def _add_strassen_2x2_block(U, V, W, start_rank, i_base, j_base, k_base):
    """
    Adds Strassen decomposition for a 2x2x2 block.
    i, j, k are local 0..1 mapped to global i_base+i, etc.
    """
    rank_offset = start_rank
    
    # Strassen 2x2 factors
    # U2, V2, W2 as defined in inspiration 2
    U2 = np.array([
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ])
    V2 = np.array([
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    W2 = np.zeros((4, 7))
    # c00 = m1 + m4 - m5 + m7
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # c01 = m3 + m5
    W2[1, 2] = 1; W2[1, 4] = 1
    # c10 = m2 + m4
    W2[2, 1] = 1; W2[2, 3] = 1
    # c11 = m1 - m2 + m3 + m6
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    for r in range(7):
        col = rank_offset + r
        # U
        for i_loc in range(2):
            for j_loc in range(2):
                # U2 row index corresponds to local (i, j)
                # In U2 definition:
                # Row 0: 00, Row 1: 01, Row 2: 10, Row 3: 11
                row_idx = i_loc * 2 + j_loc
                coeff = U2[row_idx, r]
                if coeff != 0:
                    U[(i_base + i_loc) * 5 + (j_base + j_loc), col] = coeff
        # V
        for j_loc in range(2):
            for k_loc in range(2):
                row_idx = j_loc * 2 + k_loc
                coeff = V2[row_idx, r]
                if coeff != 0:
                    V[(j_base + j_loc) * 5 + (k_base + k_loc), col] = coeff
        # W
        for k_loc in range(2):
            for i_loc in range(2):
                row_idx = k_loc * 2 + i_loc
                coeff = W2[row_idx, r]
                if coeff != 0:
                    W[(k_base + k_loc) * 5 + (i_base + i_loc), col] = coeff
                    
    return rank_offset + 7

def _add_strassen_2x2x3_block(U, V, W, start_rank, i_base, j_base, k_base):
    """
    Adds decomposition for 2x2x3 block.
    Uses Strassen for k_loc 0,1 and Trivial for k_loc 2.
    """
    rank_offset = start_rank
    
    # Part 1: k_loc 0, 1 (Strassen 2x2x2 on B's first 2 cols)
    # This is effectively a 2x2x2 multiplication where B is 2x2.
    # But B is 2x3. We only use cols 0,1.
    # The W part will only contribute to k_loc 0,1.
    
    U2 = np.array([
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ])
    V2 = np.array([
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    W2 = np.zeros((4, 7))
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    W2[1, 2] = 1; W2[1, 4] = 1
    W2[2, 1] = 1; W2[2, 3] = 1
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    for r in range(7):
        col = rank_offset + r
        # U
        for i_loc in range(2):
            for j_loc in range(2):
                row_idx = i_loc * 2 + j_loc
                coeff = U2[row_idx, r]
                if coeff != 0:
                    U[(i_base + i_loc) * 5 + (j_base + j_loc), col] = coeff
        # V
        for j_loc in range(2):
            for k_loc in range(2): # Only 0, 1
                row_idx = j_loc * 2 + k_loc
                coeff = V2[row_idx, r]
                if coeff != 0:
                    V[(j_base + j_loc) * 5 + (k_base + k_loc), col] = coeff
        # W
        for k_loc in range(2): # Only 0, 1
            for i_loc in range(2):
                row_idx = k_loc * 2 + i_loc
                coeff = W2[row_idx, r]
                if coeff != 0:
                    W[(k_base + k_loc) * 5 + (i_base + i_loc), col] = coeff
                    
    rank_offset += 7
    
    # Part 2: k_loc 2 (Trivial 2x2x1)
    # C_{i, 2} = sum_j A_{i, j} B_{j, 2}
    for i_loc in range(2):
        for j_loc in range(2):
            col = rank_offset
            U[(i_base + i_loc) * 5 + (j_base + j_loc), col] = 1.0
            V[(j_base + j_loc) * 5 + (k_base + 2), col] = 1.0
            W[(k_base + 2) * 5 + (i_base + i_loc), col] = 1.0
            rank_offset += 1
            
    return rank_offset

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Total rank estimate:
    # B1: 7
    # B2: 12
    # B3: 9
    # B4: 18
    # B5: 12
    # B6: 18
    # B7: 18
    # B8: 27
    # Total: 121
    
    total_rank = 121
    U = np.zeros((25, total_rank), dtype=np.float64)
    V = np.zeros((25, total_rank), dtype=np.float64)
    W = np.zeros((25, total_rank), dtype=np.float64)
    
    rank_offset = 0
    
    # Block 1: i 0-1, j 0-1, k 0-1 (2x2x2)
    rank_offset = _add_strassen_2x2_block(U, V, W, rank_offset, 0, 0, 0)
    
    # Block 2: i 0-1, j 2-4, k 0-1 (2x3x2)
    rank_offset = _add_trivial_block(U, V, W, rank_offset, [0, 1], [2, 3, 4], [0, 1])
    
    # Block 3: i 0-1, j 0-1, k 2-4 (2x2x3)
    rank_offset = _add_strassen_2x2x3_block(U, V, W, rank_offset, 0, 0, 2)
    
    # Block 4: i 0-1, j 2-4, k 2-4 (2x3x3)
    rank_offset = _add_trivial_block(U, V, W, rank_offset, [0, 1], [2, 3, 4], [2, 3, 4])
    
    # Block 5: i 2-4, j 0-1, k 0-1 (3x2x2)
    rank_offset = _add_trivial_block(U, V, W, rank_offset, [2, 3, 4], [0, 1], [0, 1])
    
    # Block 6: i 2-4, j 2-4, k 0-1 (3x3x2)
    rank_offset = _add_trivial_block(U, V, W, rank_offset, [2, 3, 4], [2, 3, 4], [0, 1])
    
    # Block 7: i 2-4, j 0-1, k 2-4 (3x2x3)
    rank_offset = _add_trivial_block(U, V, W, rank_offset, [2, 3, 4], [0, 1], [2, 3, 4])
    
    # Block 8: i 2-4, j 2-4, k 2-4 (3x3x3)
    rank_offset = _add_trivial_block(U, V, W, rank_offset, [2, 3, 4], [2, 3, 4], [2, 3, 4])
    
    if rank_offset != total_rank:
        # Adjust if necessary, though we calculated it
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
