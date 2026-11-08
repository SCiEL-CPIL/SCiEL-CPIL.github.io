import numpy as np
import time

DIMENSIONS = [3, 4, 6]

DEADLINE_SECONDS = 140  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 6
    
    # Total rank 66
    # U shape: (12, 66)
    # V shape: (24, 66)
    # W shape: (18, 66)
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    curr_rank = 0
    
    def add_mults(u_specs, v_specs, w_specs):
        nonlocal curr_rank
        if curr_rank >= rank:
            return
            
        # Accumulate into U, V, W
        for idx, val in u_specs:
            if val != 0:
                U[idx][curr_rank] += val
        for idx, val in v_specs:
            if val != 0:
                V[idx][curr_rank] += val
        for idx, val in w_specs:
            if val != 0:
                W[idx][curr_rank] += val
                
        curr_rank += 1

    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen Coefficients for 2x2x2
    # a: [a0, a1, a2, a3] -> A00, A01, A10, A11
    # b: [b0, b1, b2, b3] -> B00, B01, B10, B11
    # c: [c0, c1, c2, c3] -> C00, C01, C10, C11
    
    # M1: (a0+a3)(b0+b3)
    m1_a = [1, 0, 0, 1]
    m1_b = [1, 0, 0, 1]
    m1_c = [1, 0, 0, 1]
    
    # M2: (a2+a3)b0
    m2_a = [0, 0, 1, 1]
    m2_b = [1, 0, 0, 0]
    m2_c = [0, 0, 1, -1]
    
    # M3: a0(b1-b3)
    m3_a = [1, 0, 0, 0]
    m3_b = [0, 1, 0, -1]
    m3_c = [0, 1, 0, 1]
    
    # M4: a3(b2-b0)
    m4_a = [0, 0, 0, 1]
    m4_b = [-1, 0, 1, 0]
    m4_c = [1, 0, 1, 0]
    
    # M5: (a0+a1)b3
    m5_a = [1, 1, 0, 0]
    m5_b = [0, 0, 0, 1]
    m5_c = [-1, 1, 0, 0]
    
    # M6: (a2-a0)(b0+b1)
    m6_a = [-1, 0, 1, 0]
    m6_b = [1, 1, 0, 0]
    m6_c = [0, 0, 0, 1]
    
    # M7: (a1-a3)(b2+b3)
    m7_a = [0, 1, 0, -1]
    m7_b = [0, 0, 1, 1]
    m7_c = [1, 0, 0, 0]
    
    strassen_defs = [
        (m1_a, m1_b, m1_c),
        (m2_a, m2_b, m2_c),
        (m3_a, m3_b, m3_c),
        (m4_a, m4_b, m4_c),
        (m5_a, m5_b, m5_c),
        (m6_a, m6_b, m6_c),
        (m7_a, m7_b, m7_c),
    ]
    
    # Loop through the 3 blocks of P (columns of B and C)
    for k in range(3):
        if time.time() - start_time > DEADLINE_SECONDS:
            return {"U": U, "V": V, "W": W}
        
        # Base indices for B and C columns
        b_col_base = 2 * k
        c_col_base = 2 * k
        
        # 1. Schoolbook part for A[0] (1x4) * B[0:4, 2k:2k+2] (4x2)
        # This is 1x4x2. Rank 8.
        for j in range(4):
            for c_loc in range(2):
                # A[0, j] * B[j, 2k + c_loc] -> C[0, 2k + c_loc]
                
                u_spec = [(u_idx(0, j), 1)]
                v_spec = [(v_idx(j, b_col_base + c_loc), 1)]
                w_spec = [(w_idx(b_col_base + c_loc, 0), 1)]
                
                add_mults(u_spec, v_spec, w_spec)

        # 2. Strassen part for A[1:3] (2x4) * B[0:4, 2k:2k+2] (4x2)
        # Split A[1:3] into Left (cols 0,1) and Right (cols 2,3)
        # Split B into Top (rows 0,1) and Bot (rows 2,3)
        
        a_row_off = 1
        b_row_off_l = 0 # Top 2 rows of B
        b_row_off_r = 2 # Bot 2 rows of B
        
        # Left Block: A[1:3, 0:2] * B[0:2, 2k:2k+2]
        def get_a_l_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c)
            
        def get_b_l_idx(loc, row_off):
            r, c = divmod(loc, 2)
            return v_idx(row_off + r, b_col_base + c)
            
        # C local output maps to C[1:3, 2k:2k+2]
        def get_c_l_idx(loc):
            r, c = divmod(loc, 2)
            return w_idx(b_col_base + c, a_row_off + r)

        for a_c, b_c, c_c in strassen_defs:
            u_spec = [(get_a_l_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_l_idx(i, b_row_off_l), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_l_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

        # Right Block: A[1:3, 2:4] * B[2:4, 2k:2k+2]
        def get_a_r_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c + 2)
            
        for a_c, b_c, c_c in strassen_defs:
            u_spec = [(get_a_r_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_l_idx(i, b_row_off_r), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_l_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

    return {"U": U, "V": V, "W": W}
