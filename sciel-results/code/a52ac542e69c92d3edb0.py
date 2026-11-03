import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    
    # Initialize factors
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 multiplication
    # u, v, w are local indices 0..3
    # u: X (2x2), v: Y (2x2), w: Z (2x2)
    strassen_factors = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Define index mappings
    # Local index l (0..3) to Global index g
    # A (2x4): i*4 + j. Local a*2+b.
    # A1 (cols 0,1): g = a*4 + b
    # A2 (cols 2,3): g = a*4 + (b+2)
    # B (4x4): j*4 + k. Local c*2+d.
    # B1 (rows 0,1, cols 0,1): g = c*4 + d
    # B3 (rows 2,3, cols 0,1): g = (c+2)*4 + d
    # B2 (rows 0,1, cols 2,3): g = c*4 + (d+2)
    # B4 (rows 2,3, cols 2,3): g = (c+2)*4 + (d+2)
    # W (2x4): k*2 + i. Local c*2+a (where c is col, a is row).
    # W Left (cols 0,1): g = c*2 + a
    # W Right (cols 2,3): g = (c+2)*2 + a = c*2 + a + 4
    
    def add_block(a_map, b_map, w_map, col_offset):
        for r, (u_loc, v_loc, w_loc) in enumerate(strassen_factors):
            col = col_offset + r
            for l in range(4):
                if u_loc[l] != 0:
                    U[a_map[l]][col] += u_loc[l]
                if v_loc[l] != 0:
                    V[b_map[l]][col] += v_loc[l]
                if w_loc[l] != 0:
                    W[w_map[l]][col] += w_loc[l]
    
    # Mapping arrays: local index -> global index
    # A1
    a_map_1 = [0, 1, 4, 5]
    # A2
    a_map_2 = [2, 3, 6, 7]
    
    # B1
    b_map_1 = [0, 1, 4, 5]
    # B3
    b_map_3 = [8, 9, 12, 13]
    # B2
    b_map_2 = [2, 3, 6, 7]
    # B4
    b_map_4 = [10, 11, 14, 15]
    
    # W Left
    w_map_L = [0, 1, 2, 3]
    # W Right
    w_map_R = [4, 5, 6, 7]
    
    # Block 1: A1 * B1 -> W Left
    add_block(a_map_1, b_map_1, w_map_L, 0)
    
    # Block 2: A2 * B3 -> W Left
    add_block(a_map_2, b_map_3, w_map_L, 7)
    
    # Block 3: A1 * B2 -> W Right
    add_block(a_map_1, b_map_2, w_map_R, 14)
    
    # Block 4: A2 * B4 -> W Right
    add_block(a_map_2, b_map_4, w_map_R, 21)
    
    return {"U": U, "V": V, "W": W}
