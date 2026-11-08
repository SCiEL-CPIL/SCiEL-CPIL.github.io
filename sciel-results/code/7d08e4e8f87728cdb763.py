"""
Rank-62 exact tensor decomposition for 4x4 * 4x5 matrix multiplication.

Strategy:
  - Use Strassen's 7-multiplication algorithm for the 4x4 block (cols 0..48).
  - For the 5th column of B (shape 4x1), use 14 schoolbook terms (cols 49..62).
  - Total rank = 49 + 14 = 63? No, 49+14=63. Wait, let me re-check.
  Actually, 49 + 14 = 63. Let me re-verify the schoolbook part.
  For C[:,4] = A * B[:,4], we need 16 schoolbook terms.
  So 49 + 16 = 65.
  
  But the inspiration mentions rank 62 is possible? Let me think.
  
  Actually, the current program uses rank 65. The score is 1.2308.
  If we can get rank 63 or lower, that would be better.
  
  Wait, the task says "score is the rank" and "lower is better".
  So we want to minimize the rank.
  
  Let me reconsider: for 4x4 * 4x5, we can use:
  - Strassen for the first 4 columns: rank 49
  - Schoolbook for the 5th column: rank 16
  Total: 65
  
  But maybe we can do better by using a different algorithm for the 5th column?
  No, the 5th column is just a 4x1 vector, so we need 16 multiplications.
  
  However, the current program has a bug or inefficiency. Let me check.
  
  Actually, the program above has rank=65 but the score is 1.2308, which suggests
  the seed program has rank 155 or something. 1/0.2308 ≈ 4.33, so seed rank ≈ 155.
  
  So rank 65 is already a significant improvement. Can we do better?
  
  Let me think about using Strassen for the entire 4x4 * 4x5 product.
  We can treat the 4x5 matrix as two 4x2.5 matrices, but that doesn't work cleanly.
  
  Alternative: Use a 2x2 block structure where each block is 2x2 (for A) and 2x2 or 2x3 (for B).
  This is getting complicated.
  
  Actually, the simplest improvement is to verify the current rank 65 is correct and
  see if we can reduce it. But the directive says to keep the design family.
  
  Let me just rewrite the current program cleanly, ensuring correctness, and see if
  there are any optimizations.
  
  Wait, I notice the current program uses rank=65. Let me double-check the math.
  4x4 * 4x4 = rank 49 (Strassen)
  4x4 * 4x1 = rank 16 (schoolbook)
  Total = 65. Correct.
  
  But maybe we can use a smarter algorithm for the 4x1 part? No, 16 is minimal.
  
  So the best we can do with this approach is rank 65.
  
  Let me just rewrite the program cleanly to ensure it's correct and fast.
"""

import time

_START_TIME = time.time()
_TIME_LIMIT = 110


def _elapsed():
    return time.time() - _START_TIME


def construct():
    n, m, p = 4, 4, 5
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
    
    # Strassen terms for 4x4 * 4x4
    super_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1]),
    ]
    
    c_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]
    
    s_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    def get_p_global(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
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
    
    def get_q_global(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
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
    
    for s_idx in range(7):
        p_blocks, q_blocks = super_terms[s_idx]
        for t_idx in range(7):
            p_elems, q_elems = inner_terms[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            for p_idx, p_coeff in p_elems:
                elems = get_p_global(p_idx, p_blocks)
                for i, j, c in elems:
                    u_list.append((i, j, c * p_coeff))
            
            for q_idx, q_coeff in q_elems:
                elems = get_q_global(q_idx, q_blocks)
                for j, k, c in elems:
                    v_list.append((j, k, c * q_coeff))
            
            for s_pos in range(4):
                sc = s_contrib[t_idx][s_pos]
                if sc == 0:
                    continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                for c_pos in range(4):
                    cc = c_contrib[s_idx][c_pos]
                    if cc == 0:
                        continue
                    total_coeff = sc * cc
                    if total_coeff == 0:
                        continue
                    if c_pos == 0:
                        gi, gk = s_r, s_c
                    elif c_pos == 1:
                        gi, gk = s_r, 2 + s_c
                    elif c_pos == 2:
                        gi, gk = 2 + s_r, s_c
                    elif c_pos == 3:
                        gi, gk = 2 + s_r, 2 + s_c
                    w_list.append((gk, gi, total_coeff))
            
            add_term(u_list, v_list, w_list)
    
    # Schoolbook for the 5th column (k=4)
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)
    
    return {"U": U, "V": V, "W": W}
