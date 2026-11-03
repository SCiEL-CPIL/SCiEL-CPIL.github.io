"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # Winograd-style decomposition using 2x2x2 (Strassen, rank 7) and 2x2x1 (schoolbook, rank 4)
    # Split B (4x5) into B[:, 0:2] (4x2), B[:, 2:4] (4x2), B[:, 4] (4x1)
    # For each 4x2 block: split A (2x4) into A[:, 0:2] and A[:, 2:3]
    #   Each 2x2 * 2x2 uses Strassen (rank 7), so 4x2 costs 14
    # For the 4x1 block: split A into A[:, 0:2] and A[:, 2:3]
    #   Each 2x2 * 2x1 uses schoolbook (rank 4), so 4x1 costs 8
    # Total rank: 14 + 14 + 8 = 36... wait, that's the same.
    # 
    # Actually, let me reconsider. The key insight is that for 2x4x2,
    # we can potentially do better than 14 by using a combined approach.
    # But for now, let's use the same block structure but fix the bugs.
    #
    # Actually, re-examining: the current approach already does this.
    # Let me just fix the implementation to be correct.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 20 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36
    
    col = 0

    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i

        # M1 = (A00+A11)(B00+B11)
        add_rank1_term({map_a(0): 1, map_a(3): 1},
                       {map_b(0): 1, map_b(3): 1},
                       {map_c(0): 1, map_c(3): 1})

        # M2 = (A10+A11)B00
        add_rank1_term({map_a(2): 1, map_a(3): 1},
                       {map_b(0): 1},
                       {map_c(2): 1, map_c(3): -1})

        # M3 = A00(B01-B11)
        add_rank1_term({map_a(0): 1},
                       {map_b(1): 1, map_b(3): -1},
                       {map_c(1): 1, map_c(3): 1})

        # M4 = A11(B10-B00)
        add_rank1_term({map_a(3): 1},
                       {map_b(2): 1, map_b(0): -1},
                       {map_c(0): 1, map_c(2): 1})

        # M5 = (A00+A01)B11
        add_rank1_term({map_a(0): 1, map_a(1): 1},
                       {map_b(3): 1},
                       {map_c(0): -1, map_c(1): 1})

        # M6 = (A10-A00)(B00+B01)
        add_rank1_term({map_a(2): 1, map_a(0): -1},
                       {map_b(0): 1, map_b(1): 1},
                       {map_c(3): 1})

        # M7 = (A01-A11)(B10+B11)
        add_rank1_term({map_a(1): 1, map_a(3): -1},
                       {map_b(2): 1, map_b(3): 1},
                       {map_c(0): 1})

    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local) * p + b_base_j
        def map_c(local):
            i = c_base_i + local
            k = c_base_j
            return k * n + i

        # C_0 = A_00*B_0 + A_01*B_1
        # C_1 = A_10*B_0 + A_11*B_1
        add_rank1_term({map_a(0): 1}, {map_b(0): 1}, {map_c(0): 1})
        add_rank1_term({map_a(1): 1}, {map_b(1): 1}, {map_c(0): 1})
        add_rank1_term({map_a(2): 1}, {map_b(0): 1}, {map_c(1): 1})
        add_rank1_term({map_a(3): 1}, {map_b(1): 1}, {map_c(1): 1})

    # C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    # B[0:2,:] is 2x5, split into B[0:2,0:2], B[0:2,2:4], B[0:2,4]
    # B[2:4,:] is 2x5, split into B[2:4,0:2], B[2:4,2:4], B[2:4,4]

    # A[:,0:2] * B[0:2,0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # A[:,0:2] * B[0:2,2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # A[:,0:2] * B[0:2,4]
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)

    # A[:,2:4] * B[2:4,0:2]
    add_strassen_block(0, 2, 2, 0, 0, 0)
    # A[:,2:4] * B[2:4,2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2)
    # A[:,2:4] * B[2:4,4]
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
