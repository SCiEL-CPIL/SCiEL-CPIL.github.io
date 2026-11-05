import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Double Strassen for 4x4x4 (rank 49) + 4 terms for 4x4x1
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 16 x 53
    V = [[0] * rank for _ in range(m * p)]  # 20 x 53
    W = [[0] * rank for _ in range(p * n)]  # 20 x 53
    
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
    
    # --- Part 1: 4x4 * 4x4 using Double Strassen (49 terms) ---
    
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)])
    ]
    
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)])
    ]
    
    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:    # A00
                gi, gj = r, c
            elif b_idx == 1:  # A01
                gi, gj = r, 2 + c
            elif b_idx == 2:  # A10
                gi, gj = 2 + r, c
            elif b_idx == 3:  # A11
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
            if b_idx == 0:    # B00
                gj, gk = r, c
            elif b_idx == 1:  # B01
                gj, gk = r, 2 + c
            elif b_idx == 2:  # B10
                gj, gk = 2 + r, c
            elif b_idx == 3:  # B11
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
    
    # --- Part 2: 4x4 * 4x1 using rank-4 decomposition ---
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # For each output row i, we need one term:
    #   U: A[i,:] all ones (for fixed i)
    #   V: B[:,4] all ones (for fixed i)  
    #   W: C[4,i] = 1
    for i in range(4):
        u_list = []
        v_list = []
        w_list = []
        
        # U: all elements in row i of A
        for j in range(4):
            u_list.append((i, j, 1))
        
        # V: all elements in column 4 of B
        for j in range(4):
            v_list.append((j, 4, 1))
        
        # W: output at position (k=4, i=i)
        w_list.append((4, i, 1))
        
        add_term(col, u_list, v_list, w_list)
        col += 1
    
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}
