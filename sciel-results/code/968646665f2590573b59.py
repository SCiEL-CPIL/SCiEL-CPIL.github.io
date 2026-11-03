def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 8 x 22
    V = [[0] * rank for _ in range(m * p)]  # 20 x 22
    W = [[0] * rank for _ in range(p * n)]  # 10 x 22
    
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
        
        terms = [
            ({map_a(0): 1, map_a(3): 1}, {map_b(0): 1, map_b(3): 1}, {map_c(0): 1, map_c(3): 1}),
            ({map_a(2): 1, map_a(3): 1}, {map_b(0): 1}, {map_c(2): 1, map_c(3): -1}),
            ({map_a(0): 1}, {map_b(1): 1, map_b(3): -1}, {map_c(1): 1, map_c(3): 1}),
            ({map_a(3): 1}, {map_b(2): 1, map_b(0): -1}, {map_c(0): 1, map_c(2): 1}),
            ({map_a(0): 1, map_a(1): 1}, {map_b(3): 1}, {map_c(0): -1, map_c(1): 1}),
            ({map_a(2): 1, map_a(0): -1}, {map_b(0): 1, map_b(1): 1}, {map_c(3): 1}),
            ({map_a(1): 1, map_a(3): -1}, {map_b(2): 1, map_b(3): 1}, {map_c(0): 1}),
        ]
        for u, v, w in terms:
            add_rank1_term(u, v, w)
    
    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local) * p + b_base_j
        def map_c(local):
            i = c_base_i + local
            k = c_base_j
            return k * n + i
        
        for l in range(4):
            a_local = l
            b_local = l % 2
            c_local = l // 2
            u = {map_a(a_local): 1}
            v = {map_b(b_local): 1}
            w = {map_c(c_local): 1}
            add_rank1_term(u, v, w)
    
    def add_rank2_term_2x1_1x5(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # A is 2x1 at (a_base_i, a_base_j), B is 1x5 at (b_base_i, b_base_j)
        # C[i,k] = A[i,0] * B[0,k]
        # Rank 2: one term for each i
        for i in range(2):
            u = {(a_base_i + i) * m + a_base_j: 1}
            v = {}
            w = {}
            for k in range(5):
                b_idx = b_base_i * p + (b_base_j + k)
                c_idx = (c_base_j + k) * n + (c_base_i + i)
                v[b_idx] = 1
                w[c_idx] = 1
            add_rank1_term(u, v, w)
    
    # A (2x4) split into A1 (2x3) cols 0-2, A2 (2x1) col 3
    # B (4x5) split into B1 (3x5) rows 0-2, B2 (1x5) row 3
    
    # A1 (2x3) x B1 (3x5):
    #   A1a (2x2) cols 0-1, A1b (2x1) col 2
    #   B1a (2x5) rows 0-1, B1b (1x5) row 2
    
    # A1a (2x2) x B1a (2x5):
    #   B1a1 (2x2) cols 0-1, B1a2 (2x2) cols 2-3, B1a3 (2x1) col 4
    add_strassen_block(0, 0, 0, 0, 0, 0)    # A1a x B1a1
    add_strassen_block(0, 0, 0, 2, 0, 2)    # A1a x B1a2
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)  # A1a x B1a3
    
    # A1b (2x1) x B1b (1x5):
    add_rank2_term_2x1_1x5(0, 2, 2, 0, 0, 0)
    
    # A2 (2x1) x B2 (1x5):
    add_rank2_term_2x1_1x5(0, 3, 3, 0, 0, 0)
    
    return {"U": U, "V": V, "W": W}
