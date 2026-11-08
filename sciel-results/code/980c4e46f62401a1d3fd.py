import numpy as np

def construct():
    """
    Returns a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Rank: 114 (49 for 4x4*4x4, 49 for 4x4*4x4, 16 for 4x4*4x1)
    """
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Strassen 2x2x2 terms
    # (A_indices, B_indices, C_indices)
    # Indices are in 0..3 for 2x2 flattened (row-major: 0,1,2,3 -> 00,01,10,11)
    # Actually, standard Strassen for 2x2:
    # m1 = (A00+A11)(B00+B11) -> C00 += m1, C11 += m1
    # m2 = (A10+A11)B00 -> C10 += m2, C11 -= m2
    # m3 = A00(B01+B11) -> C01 += m3, C00 -= m3? No.
    # Let's use the standard 7 terms explicitly.
    
    s2x2 = [
        # (A_mask, B_mask, C_mask)
        # A_mask: dict {idx: coeff}, B_mask: dict, C_mask: dict
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def strassen_4x4(block_offset_k):
        """
        Adds Strassen decomposition for 4x4 * 4x4 * 4.
        block_offset_k: offset for the B and C columns (0, 4, etc.)
        """
        # 4x4 is composed of four 2x2 blocks.
        # We apply Strassen to the 2x2 block matrix multiplication.
        # This requires 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # Total 49.
        
        # The 7 terms of 2x2 block Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01+B11)
        # M4 = A11(B10+B00)
        # M5 = (A00+A01)B11
        # M6 = (A10+A00)(B00+B01)
        # M7 = (A01+A11)(B10+B11)
        
        # Indices in 2x2 block space:
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 10, 11
        # C blocks: 00, 01, 10, 11
        
        block_mults = [
            # (A_blocks, B_blocks, C_blocks)
            # Each is a list of (block_idx, coeff)
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], {(0): 1, (3): 1}),
            ([(2, 1), (3, 1)], [(0, 1)], {(2): 1, (3): -1}),
            ([(0, 1)], [(1, 1), (3, -1)], {(1): 1, (3): 1}),
            ([(3, 1)], [(2, 1), (0, -1)], {(0): 1, (2): 1}),
            ([(0, 1), (1, 1)], [(3, 1)], {(0): -1, (1): 1}),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], {(3): 1}),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], {(0): 1}),
        ]
        
        for a_blocks, b_blocks, c_blocks in block_mults:
            for a_meta, b_meta, c_meta in s2x2:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # A: 4x4. Blocks are 2x2.
                # Global indices: i, j
                # Block (bi, bj), local (li, lj)
                # Global i = bi*2 + li, j = bj*2 + lj
                for local_idx, coeff in a_meta.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for b_idx, b_coeff in a_blocks:
                        bi = b_idx // 2
                        bj = b_idx % 2
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        u_vec[gi * m + gj] += coeff * b_coeff
                
                # B: 4x4 (subset of 4x9).
                # Global indices: j, k
                # Block (bi, bj) in B space?
                # B is 4x4 here.
                # j = bi*2 + li, k = bj*2 + lj
                # Note: block_offset_k adds to k
                for local_idx, coeff in b_meta.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for b_idx, b_coeff in b_blocks:
                        bi = b_idx // 2
                        bj = b_idx % 2
                        gj = bi * 2 + li
                        gk = bj * 2 + lj + block_offset_k
                        v_vec[gj * p + gk] += coeff * b_coeff
                
                # C: 4x4 (subset of 4x9).
                # Global indices: i, k
                # Block (bi, bj) in C space?
                # i = bi*2 + li, k = bj*2 + lj
                # Note: block_offset_k adds to k
                for local_idx, coeff in c_meta.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for c_idx, c_coeff in c_blocks.items():
                        bi = c_idx // 2
                        bj = c_idx % 2
                        gi = bi * 2 + li
                        gk = bj * 2 + lj + block_offset_k
                        w_vec[gk * n + gi] += coeff * c_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B[0:4] (4x4 * 4x4)
    strassen_4x4(0)
    
    # Part 2: A * B[4:8] (4x4 * 4x4)
    strassen_4x4(4)
    
    # Part 3: A * B[8:9] (4x4 * 4x1)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
