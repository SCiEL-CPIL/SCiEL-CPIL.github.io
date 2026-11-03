import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # The rank of 2x4 by 4x8 matrix multiplication is 56.
    # This is achieved by tiling into 2x2 by 2x2 multiplications using Strassen's algorithm.
    # A (2x4) is split into two 2x2 blocks: A_left, A_right.
    # B (4x8) is split into two 2x8 blocks: B_top, B_bottom.
    # B_top and B_bottom are further split into four 2x2 blocks each.
    # C = A_left @ B_top + A_right @ B_bottom.
    # Each 2x2 @ 2x2 @ 2x2 uses 7 scalar multiplications.
    # There are 4 such blocks in B_top and 4 in B_bottom.
    # Total rank = 8 * 7 = 56.

    # Strassen 2x2 local factorization (rank 7).
    # U_loc[local_a_idx][r], V_loc[local_b_idx][r], W_loc[local_c_idx][r]
    # local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ]

    total_rank = 56
    num_U = n * m  # 8
    num_V = m * p  # 32
    num_W = p * n  # 16

    U = [[0] * total_rank for _ in range(num_U)]
    V = [[0] * total_rank for _ in range(num_V)]
    W = [[0] * total_rank for _ in range(num_W)]

    # Group 0: A_left (cols 0,1) @ B_top (rows 0,1)
    # Group 1: A_right (cols 2,3) @ B_bottom (rows 2,3)
    # For each group, we have 4 column blocks of B (each 2 cols wide).
    
    for group in range(2):
        # Determine A columns and B rows for this group
        if group == 0:
            a_col_offset = 0
            b_row_offset = 0
        else:
            a_col_offset = 2
            b_row_offset = 2
            
        for jj in range(p // 2): # 4 column blocks
            base_r = (group * 4 + jj) * 7
            
            # Map local A indices to global U indices
            # A is 2x4. Row i, Col j. Global idx = i*m + j.
            # Local a00 -> (0, a_col_offset)
            # Local a01 -> (0, a_col_offset+1)
            # Local a10 -> (1, a_col_offset)
            # Local a11 -> (1, a_col_offset+1)
            a_map = [
                0 * m + (a_col_offset + 0),
                0 * m + (a_col_offset + 1),
                1 * m + (a_col_offset + 0),
                1 * m + (a_col_offset + 1)
            ]
            
            # Map local B indices to global V indices
            # B is 4x8. Row j, Col k. Global idx = j*p + k.
            # Local b00 -> (b_row_offset, 2*jj)
            # Local b01 -> (b_row_offset, 2*jj+1)
            # Local b10 -> (b_row_offset+1, 2*jj)
            # Local b11 -> (b_row_offset+1, 2*jj+1)
            b_map = [
                (b_row_offset + 0) * p + (2 * jj + 0),
                (b_row_offset + 0) * p + (2 * jj + 1),
                (b_row_offset + 1) * p + (2 * jj + 0),
                (b_row_offset + 1) * p + (2 * jj + 1)
            ]
            
            # Map local C indices to global W indices
            # C is 2x8. Row i, Col k. W convention: W[k*n + i, r].
            # Local c00 -> (0, 2*jj) -> W idx: (2*jj)*2 + 0
            # Local c01 -> (0, 2*jj+1) -> W idx: (2*jj+1)*2 + 0
            # Local c10 -> (1, 2*jj) -> W idx: (2*jj)*2 + 1
            # Local c11 -> (1, 2*jj+1) -> W idx: (2*jj+1)*2 + 1
            c_map = [
                (2 * jj + 0) * n + 0,
                (2 * jj + 1) * n + 0,
                (2 * jj + 0) * n + 1,
                (2 * jj + 1) * n + 1
            ]
            
            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la][r]
                    if val:
                        U[a_map[la]][rr] += val
                for lb in range(4):
                    val = V_loc[lb][r]
                    if val:
                        V[b_map[lb]][rr] += val
                for lc in range(4):
                    val = W_loc[lc][r]
                    if val:
                        W[c_map[lc]][rr] += val

    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
