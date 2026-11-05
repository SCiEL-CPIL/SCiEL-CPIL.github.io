import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit

    # We'll use a block decomposition:
    # A is 4x4, B is 4x5, C is 4x5
    # Split B into B1 (4x4) and B2 (4x1)
    # C = A*B1 + A*B2
    # A*B1: 4x4 * 4x4 -> 4x4, use Strassen (rank 49) or Winograd F(4) (rank 40)
    # A*B2: 4x4 * 4x1 -> 4x1, schoolbook (rank 16)
    # Total: 49 + 16 = 65 or 40 + 16 = 56

    # Let me implement the Winograd F(4) approach carefully.
    # The F(4) transform for 4x4 matrix multiplication uses rank 40.
    
    # Actually, let me use a simpler but correct approach:
    # Use the 49-term Strassen for 4x4x4 and 16-term schoolbook for 4x4x1.
    # Total rank = 65.
    
    # But wait - the current program already achieves score 1.2308 with rank 65.
    # Let me see if I can do better.
    
    # Actually, looking at the problem more carefully:
    # The 4x4x4 part can be done with rank 40 using Winograd F(4).
    # Let me try to implement that.
    
    # F(4) transform matrices:
    # Row transform F_row (16x4):
    F_row = np.array([
        [1, 0, 0, 0],   # 0
        [0, 0, 0, 1],   # 1
        [1, 0, 0, 1],   # 2
        [0, 1, -1, 0],  # 3
        [0, 1, 1, 0],   # 4
        [0, 1, 1, 0],   # 5
        [0, 1, 1, 0],   # 6
        [0, 1, 0, 0],   # 7
        [0, 1, 0, 0],   # 8
        [0, 1, 0, 0],   # 9
        [0, 0, 1, 0],   # 10
        [0, 0, 1, 0],   # 11
        [0, 0, 1, 0],   # 12
        [1, 0, 0, -1],  # 13
        [1, 0, 0, -1],  # 14
        [1, 0, 0, -1],  # 15
    ])
    
    F_col = np.array([
        [1, 0, 0, 0],   # 0
        [0, 0, 0, 1],   # 1
        [1, 0, 0, 1],   # 2
        [0, 1, -1, 0],  # 3
        [0, 1, 1, 0],   # 4
        [0, 1, 1, 0],   # 5
        [0, 1, 1, 0],   # 6
        [0, 1, 0, 0],   # 7
        [0, 1, 0, 0],   # 8
        [0, 1, 0, 0],   # 9
        [0, 0, 1, 0],   # 10
        [0, 0, 1, 0],   # 11
        [0, 0, 1, 0],   # 12
        [1, 0, 0, -1],  # 13
        [1, 0, 0, -1],  # 14
        [1, 0, 0, -1],  # 15
    ])
    
    # For the tensor decomposition, I need to figure out the 40 terms.
    # This is complex. Let me just use the safe 65-rank approach.
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Strassen 2x2x2 decomposition
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),       # M1
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),             # M2
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),             # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),             # M4
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),             # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),               # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),               # M7
    ]

    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),       # M1 -> C00, C11
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),      # M2 -> C10, -C11
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),      # M3 -> C01, C11
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),      # M4 -> C00, C10
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),      # M5 -> -C00, C01
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),              # M6 -> C11
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),              # M7 -> C00
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))

            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))

            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 (rank 16) - schoolbook
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    if time.time() - start_time > DEADLINE - 5:
        pass

    return {"U": U, "V": V, "W": W}
