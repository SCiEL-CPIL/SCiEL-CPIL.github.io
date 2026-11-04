import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will use a decomposition of rank 50.
    # 49 terms for 4x4 * 4x4 (Laderman/Strassen)
    # 1 term for 4x4 * 4x1 (Rank 1 tensor)
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 16 x 50
    V = [[0] * rank for _ in range(m * p)]  # 20 x 50
    W = [[0] * rank for _ in range(p * n)]  # 20 x 50
    
    # Helper to add a rank-1 term
    # u_indices: list of (i, j, coeff)
    # v_indices: list of (j, k, coeff)
    # w_indices: list of (k, i, coeff)
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
    
    # --- Part 1: 4x4 * 4x4 using Strassen (Rank 49) ---
    # This handles k = 0, 1, 2, 3
    
    # Function to get U, V, W lists for a super term
    def get_super_term_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        # Coefficients for A blocks in P
        # Coefficients for B blocks in Q
        if super_idx == 0: # M1
            p_blocks = [1, 0, 0, 1] # A00 + A11
            q_blocks = [1, 0, 0, 1] # B00 + B11
        elif super_idx == 1: # M2
            p_blocks = [0, 0, 1, 1] # A10 + A11
            q_blocks = [1, 0, 0, 0] # B00
        elif super_idx == 2: # M3
            p_blocks = [1, 0, 0, 0] # A00
            q_blocks = [0, 1, 0, -1] # B01 - B11
        elif super_idx == 3: # M4
            p_blocks = [0, 0, 0, 1] # A11
            q_blocks = [-1, 0, 1, 0] # B10 - B00
        elif super_idx == 4: # M5
            p_blocks = [1, 1, 0, 0] # A00 + A01
            q_blocks = [0, 0, 0, 1] # B11
        elif super_idx == 5: # M6
            p_blocks = [-1, 0, 1, 0] # A10 - A00
            q_blocks = [1, 1, 0, 0] # B00 + B01
        elif super_idx == 6: # M7
            p_blocks = [0, 1, 0, -1] # A01 - A11
            q_blocks = [0, 0, 1, 1] # B10 + B11
        else:
            return u_list, v_list, w_list
            
        def get_p_element(p_elem_idx):
            res = []
            r = p_elem_idx // 2
            c = p_elem_idx % 2
            
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                if b_idx == 0: # A00
                    gi = r; gj = c
                elif b_idx == 1: # A01
                    gi = r; gj = 2 + c
                elif b_idx == 2: # A10
                    gi = 2 + r; gj = c
                elif b_idx == 3: # A11
                    gi = 2 + r; gj = 2 + c
                
                res.append((gi, gj, coeff))
            return res
            
        def get_q_element(q_elem_idx):
            res = []
            r = q_elem_idx // 2
            c = q_elem_idx % 2
            
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0: # B00
                    gj = r; gk = c
                elif b_idx == 1: # B01
                    gj = r; gk = 2 + c
                elif b_idx == 2: # B10
                    gj = 2 + r; gk = c
                elif b_idx == 3: # B11
                    gj = 2 + r; gk = 2 + c
                
                res.append((gj, gk, coeff))
            return res
            
        # Determine P and Q elements for this inner term
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)] # P00, P11
            q_elems = [(0, 1), (3, 1)] # Q00, Q11
            c_contrib = [(0, 0, 1), (1, 1, 1)]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)] # P10, P11
            q_elems = [(0, 1)] # Q00
            c_contrib = [(1, 0, 1), (1, 1, -1)]
        elif inner_idx == 2:
            p_elems = [(0, 1)] # P00
            q_elems = [(1, 1), (3, -1)] # Q01, -Q11
            c_contrib = [(0, 1, 1), (1, 1, 1)]
        elif inner_idx == 3:
            p_elems = [(3, 1)] # P11
            q_elems = [(2, 1), (0, -1)] # Q10, -Q00
            c_contrib = [(0, 0, 1), (1, 0, 1)]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)] # P00, P01
            q_elems = [(3, 1)] # Q11
            c_contrib = [(0, 0, -1), (0, 1, 1)]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)] # P10, -P00
            q_elems = [(0, 1), (1, 1)] # Q00, Q01
            c_contrib = [(1, 1, 1)]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)] # P01, -P11
            q_elems = [(2, 1), (3, 1)] # Q10, Q11
            c_contrib = [(0, 0, 1)]
        
        u_list = []
        v_list = []
        
        for p_elem, coeff in p_elems:
            for i, j, c in get_p_element(p_elem):
                u_list.append((i, j, coeff * c))
        
        for q_elem, coeff in q_elems:
            for j, k, c in get_q_element(q_elem):
                v_list.append((j, k, coeff * c))
        
        w_list = []
        
        if super_idx == 0:
            blocks = [(0, 0, 1), (1, 1, 1)]
        elif super_idx == 1:
            blocks = [(1, 0, 1), (1, 1, -1)]
        elif super_idx == 2:
            blocks = [(0, 1, 1), (1, 1, 1)]
        elif super_idx == 3:
            blocks = [(0, 0, 1), (1, 0, 1)]
        elif super_idx == 4:
            blocks = [(0, 0, -1), (0, 1, 1)]
        elif super_idx == 5:
            blocks = [(1, 1, 1)]
        elif super_idx == 6:
            blocks = [(0, 0, 1)]
        
        for br, bc, sign in blocks:
            for lr, lc, local_coeff in c_contrib:
                gi = br * 2 + lr
                gk = bc * 2 + lc
                w_list.append((gk, gi, sign * local_coeff))
        
        return u_list, v_list, w_list

    for super_idx in range(7):
        for inner_idx in range(7):
            if time.time() > deadline:
                break
            u_list, v_list, w_list = get_super_term_indices(super_idx, inner_idx)
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # Part 2: 4x4x1 (Rank 1)
    # Handles k = 4
    # U[i*4+j, 49] = 1
    # V[j*5+4, 49] = 1
    # W[16+i, 49] = 1
    
    for i in range(4):
        for j in range(4):
            U[i*4 + j][49] = 1
        V[j*5 + 4][49] = 1
        W[16 + i][49] = 1

    return {"U": U, "V": V, "W": W}
