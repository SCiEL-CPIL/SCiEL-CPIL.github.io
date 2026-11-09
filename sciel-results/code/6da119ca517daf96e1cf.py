"""
Optimized rank-65 decomposition for 4x4x5 matrix multiplication.
Uses Strassen (rank 49) for the first 4 columns and Schoolbook (rank 16) for the 5th.
"""

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Theoretical Rank:
    # M_4,4,5 = M_4,4,4 (rank 49) + M_4,4,1 (rank 16)
    # Total Rank = 65
    rank = 65
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for i, j, c in u_entries:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_entries:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_entries:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
        
    # Part 1: Strassen for 4x4 * 4x4 (first 4 columns of B)
    # We use the standard 4x4 Strassen decomposition (rank 49)
    # Defined by 7 super-terms and 7 inner-terms
    
    # Super terms (M1..M7) for 4x4 blocks
    # Format: (P_blocks, Q_blocks)
    # P_blocks: coeffs for A00, A01, A10, A11
    # Q_blocks: coeffs for B00, B01, B10, B11
    super_defs = [
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1])  # M7
    ]
    
    # Inner terms (S1..S7) for 2x2 blocks
    # Format: (P_elems, Q_elems)
    # P_elems: list of (elem_idx, coeff) where elem_idx 0=P00, 1=P01, 2=P10, 3=P11
    inner_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]), # S1
        ([(2, 1), (3, 1)], [(0, 1)]),          # S2
        ([(0, 1)], [(1, 1), (3, -1)]),         # S3
        ([(3, 1)], [(2, 1), (0, -1)]),         # S4
        ([(0, 1), (1, 1)], [(3, 1)]),          # S5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]), # S6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])  # S7
    ]
    
    # Contributions to 2x2 result blocks
    inner_contribs = [
        [1, 0, 0, 1], # S1
        [0, 0, 1, -1], # S2
        [0, 1, 0, 1], # S3
        [1, 0, 1, 0], # S4
        [-1, 1, 0, 0], # S5
        [0, 0, 0, 1], # S6
        [1, 0, 0, 0]  # S7
    ]
    
    # Contributions to 4x4 result blocks
    super_contribs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    for s_idx in range(7):
        p_blocks, q_blocks = super_defs[s_idx]
        s_contribs = super_contribs[s_idx]
        
        for t_idx in range(7):
            p_elems, q_elems = inner_defs[t_idx]
            t_contribs = inner_contribs[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            # Expand U
            for p_idx, p_coeff in p_elems:
                pr = p_idx // 2
                pc = p_idx % 2
                for b_idx, b_coeff in enumerate(p_blocks):
                    if b_coeff == 0: continue
                    base_r = (b_idx // 2) * 2
                    base_c = (b_idx % 2) * 2
                    gi = base_r + pr
                    gj = base_c + pc
                    u_list.append((gi, gj, b_coeff * p_coeff))
                    
            # Expand V (only for first 4 columns of B, k=0..3)
            for q_idx, q_coeff in q_elems:
                qr = q_idx // 2
                qc = q_idx % 2
                for b_idx, b_coeff in enumerate(q_blocks):
                    if b_coeff == 0: continue
                    base_r = (b_idx // 2) * 2
                    base_c = (b_idx % 2) * 2
                    gj = base_r + qr
                    gk = base_c + qc
                    v_list.append((gj, gk, b_coeff * q_coeff))
                    
            # Expand W
            for s_pos in range(4):
                t_c = t_contribs[s_pos]
                if t_c == 0: continue
                sr = s_pos // 2
                sc = s_pos % 2
                for c_block in range(4):
                    s_c = s_contribs[c_block]
                    if s_c == 0: continue
                    total_coeff = t_c * s_c
                    if total_coeff == 0: continue
                    base_i = (c_block // 2) * 2
                    base_k = (c_block % 2) * 2
                    gi = base_i + sr
                    gk = base_k + sc
                    w_list.append((gk, gi, total_coeff))
                    
            add_term(u_list, v_list, w_list)

    # Part 2: Schoolbook for column 4 (k=4)
    for i in range(n):
        for j in range(m):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
