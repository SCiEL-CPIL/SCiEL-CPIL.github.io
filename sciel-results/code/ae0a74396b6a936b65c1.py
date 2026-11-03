"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # We use a decomposition of rank 36.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3)
    # Split B into B1 (rows 0,1) and B2 (rows 2,3)
    # C = A1*B1 + A2*B2
    
    # Each block is 2x2 * 2x5
    # Split B_block into B_col0_1 (2x2), B_col2_3 (2x2), B_col4 (2x1)
    # Rank 7 + 7 + 4 = 18 per block.
    # Total rank 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 20 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36
    
    col = 0
    
    # Helper to add a rank-1 term
    # a_idx: index in U (i*m+j)
    # b_idx: index in V (j*p+k)
    # c_idx: index in W (k*n+i)
    # coeff: value
    def add_term(a_idx, b_idx, c_idx, coeff):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[a_idx][col] += coeff
        V[b_idx][col] += coeff
        W[c_idx][col] += coeff
        col += 1

    # Implementation of adding a rank-1 term defined by sparse dicts
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

    # Define Strassen for a specific block
    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # a_base_i, a_base_j: global start index of A block (2x2)
        # b_base_i, b_base_j: global start index of B block (2x2)
        # c_base_i, c_base_j: global start index of C block (2x2)
        
        # Local indices 0,1,2,3 for 2x2 matrix
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            # W index is k*n + i.
            # C local (i, j) -> global i = c_base_i + i, k = c_base_j + j
            # Index = k * n + i
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i

        # M1
        u = {map_a(0): 1, map_a(3): 1}
        v = {map_b(0): 1, map_b(3): 1}
        w = {map_c(0): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        # M2
        u = {map_a(2): 1, map_a(3): 1}
        v = {map_b(0): 1}
        w = {map_c(2): 1, map_c(3): -1}
        add_rank1_term(u, v, w)

        # M3
        u = {map_a(0): 1}
        v = {map_b(1): 1, map_b(3): -1}
        w = {map_c(1): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        # M4
        u = {map_a(3): 1}
        v = {map_b(2): 1, map_b(0): -1}
        w = {map_c(0): 1, map_c(2): 1}
        add_rank1_term(u, v, w)

        # M5
        u = {map_a(0): 1, map_a(1): 1}
        v = {map_b(3): 1}
        w = {map_c(0): -1, map_c(1): 1}
        add_rank1_term(u, v, w)

        # M6
        u = {map_a(2): 1, map_a(0): -1}
        v = {map_b(0): 1, map_b(1): 1}
        w = {map_c(3): 1}
        add_rank1_term(u, v, w)

        # M7
        u = {map_a(1): 1, map_a(3): -1}
        v = {map_b(2): 1, map_b(3): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)

    # Schoolbook for 2x2 * 2x1
    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # B is 2x1. Local B indices 0, 1.
        # C is 2x1. Local C indices 0, 1.
        
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            # B is 2x1. p=5.
            # Global B index: row * p + col
            return (b_base_i + local) * p + b_base_j
        def map_c(local):
            # C is 2x1.
            # Global C index: k * n + i
            i = c_base_i + local
            k = c_base_j
            return k * n + i

        # 4 terms
        for l in range(4):
            # A local l, B local l%2? No.
            # C_i = sum_j A_ij B_j
            # Term for A_00 B_0 -> C_0
            # Term for A_01 B_1 -> C_0
            # Term for A_10 B_0 -> C_1
            # Term for A_11 B_1 -> C_1
            
            a_local = l
            b_local = l % 2
            c_local = l // 2
            
            u = {map_a(a_local): 1}
            v = {map_b(b_local): 1}
            w = {map_c(c_local): 1}
            add_rank1_term(u, v, w)

    # Block 1: A1 (rows 0,1; cols 0,1), B1 (rows 0,1; cols 0,4)
    # B1 is 2x5. Split into B1_01 (cols 0,1), B1_23 (cols 2,3), B1_4 (col 4)
    
    # A1 base: i=0, j=0
    # B1 base: i=0, j=0
    # C base: i=0, j=0
    
    add_strassen_block(0, 0, 0, 0, 0, 0) # A1 * B1_01
    add_strassen_block(0, 0, 0, 2, 0, 2) # A1 * B1_23
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4) # A1 * B1_4
    
    # Block 2: A2 (rows 0,1; cols 2,3), B2 (rows 2,3; cols 0,4)
    # A2 base: i=0, j=2
    # B2 base: i=2, j=0
    # C base: i=0, j=0
    
    add_strassen_block(0, 2, 2, 0, 0, 0) # A2 * B2_01
    add_strassen_block(0, 2, 2, 2, 0, 2) # A2 * B2_23
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4) # A2 * B2_4
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5:
        # Should not happen for this static construction, but safety first
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
