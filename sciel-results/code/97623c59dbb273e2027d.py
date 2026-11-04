import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimation:
    # 2 blocks of (2x5x6)
    # Each (2x5x6) splits into (2x2x6) + (2x3x6)
    # (2x2x6) = 3 * (2x2x2) Strassen (rank 7 each) -> 21
    # (2x3x6) = 3 * (2x3x2) Schoolbook (rank 12 each) -> 36
    # Total per (2x5x6) = 57
    # Total Rank = 2 * 57 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0

    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal current_rank
        r = current_rank
        current_rank += 1
        for idx in u_indices:
            U[idx][r] += 1
        for idx in v_indices:
            V[idx][r] += 1
        for idx, val in w_indices:
            W[idx][r] += val

    # Helper for Strassen 2x2x2
    # Local A indices: 0,1 (row 0), 2,3 (row 1)
    # Local B indices: 0,1 (col 0), 2,3 (col 1)
    # Local C indices: 0,1 (col 0), 2,3 (col 1)
    def add_strassen_2x2x2(a_base, b_base, c_base):
        # a_base: global index of A[0,0] in this block
        # b_base: global index of B[0,0] in this block
        # c_base: global index of C[0,0] in this block
        
        # M1: (A00+A11)(B00+B11) -> C00+, C11+
        add_term([a_base, a_base+3], [b_base, b_base+3], [(c_base, 1), (c_base+3, 1)])
        
        # M2: (A10+A11)B00 -> C10+, C11-
        add_term([a_base+2, a_base+3], [b_base], [(c_base+2, 1), (c_base+3, -1)])
        
        # M3: A00(B01-B11) -> C01+, C11+
        add_term([a_base], [b_base+1, b_base+3], [(c_base+1, 1), (c_base+3, 1)])
        
        # M4: A11(B10-B00) -> C00+, C10+
        add_term([a_base+3], [b_base+2, b_base], [(c_base, 1), (c_base+2, 1)])
        
        # M5: (A00+A01)B11 -> C00-, C01+
        add_term([a_base, a_base+1], [b_base+3], [(c_base, -1), (c_base+1, 1)])
        
        # M6: (A10-A00)(B00+B01) -> C11+
        add_term([a_base+2, a_base], [b_base, b_base+1], [(c_base+3, 1)])
        
        # M7: (A01-A11)(B10+B11) -> C00+
        add_term([a_base+1, a_base+3], [b_base+2, b_base+3], [(c_base, 1)])

    # Helper for Schoolbook 2x3x2
    # Local A: 2x3 (indices 0..5)
    # Local B: 3x2 (indices 0..5)
    # Local C: 2x2 (indices 0..3)
    def add_schoolbook_2x3x2(a_base, b_base, c_base):
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    a_idx = a_base + i * 3 + j
                    b_idx = b_base + j * 2 + k
                    c_idx = c_base + i * 2 + k
                    add_term([a_idx], [b_idx], [(c_idx, 1)])

    # Process Top Half of A (Rows 0-1)
    # A is 4x5. Top is 2x5.
    # B is 5x6.
    # C is 6x4. Top C is 2x6.
    
    # Split A_top (2x5) into A_top1 (2x2, cols 0-1) and A_top2 (2x3, cols 2-4)
    # Split B (5x6) into B1 (2x6, rows 0-1) and B2 (3x6, rows 2-4)
    
    # Part 1: A_top1 (2x2) * B1 (2x6)
    # B1 is 2x6. Split into 3 blocks of 2x2.
    # Block 1.1: A[0:2, 0:2] * B[0:2, 0:2]
    # A base: 0. B base: 0. C base: 0 (C[0:2, 0:2] -> indices 0,1,2,3 in C? 
    # C is 6x4. C[i,k]. i=0,1. k=0,1.
    # W index: k*n + i.
    # C[0,0] -> 0*4+0=0. C[0,1] -> 1*4+0=4. C[1,0] -> 0*4+1=1. C[1,1] -> 1*4+1=5.
    # Wait, my local C indices in add_strassen are 0,1,2,3 corresponding to C00, C01, C10, C11.
    # In global W, C00 is idx 0, C01 is idx 4, C10 is idx 1, C11 is idx 5.
    # So I need to map local C indices to global W indices.
    
    # Let's define a mapping function for C indices in the 2x2x2 block.
    # Local i, k -> Global i, k.
    # Global W index = k * n + i.
    
    def get_w_index(g_i, g_k):
        return g_k * n + g_i

    # Re-define add_strassen to take global offsets
    def add_strassen_2x2x2_global(a_g_i, a_g_j, b_g_j, b_g_k, c_g_i, c_g_k):
        # a_g_i, a_g_j: top-left of A block
        # b_g_j, b_g_k: top-left of B block
        # c_g_i, c_g_k: top-left of C block
        
        # Local A indices
        a00 = a_g_i * m + a_g_j
        a01 = a_g_i * m + a_g_j + 1
        a10 = (a_g_i + 1) * m + a_g_j
        a11 = (a_g_i + 1) * m + a_g_j + 1
        
        # Local B indices
        b00 = b_g_j * p + b_g_k
        b01 = b_g_j * p + b_g_k + 1
        b10 = (b_g_j + 1) * p + b_g_k
        b11 = (b_g_j + 1) * p + b_g_k + 1
        
        # Local C indices
        c00 = get_w_index(c_g_i, c_g_k)
        c01 = get_w_index(c_g_i, c_g_k + 1)
        c10 = get_w_index(c_g_i + 1, c_g_k)
        c11 = get_w_index(c_g_i + 1, c_g_k + 1)
        
        # M1
        add_term([a00, a11], [b00, b11], [(c00, 1), (c11, 1)])
        # M2
        add_term([a10, a11], [b00], [(c10, 1), (c11, -1)])
        # M3
        add_term([a00], [b01, b11], [(c01, 1), (c11, 1)])
        # M4
        add_term([a11], [b10, b00], [(c00, 1), (c10, 1)])
        # M5
        add_term([a00, a01], [b11], [(c00, -1), (c01, 1)])
        # M6
        add_term([a10, a00], [b00, b01], [(c11, 1)])
        # M7
        add_term([a01, a11], [b10, b11], [(c00, 1)])

    # Re-define add_schoolbook_2x3x2_global
    def add_schoolbook_2x3x2_global(a_g_i, a_g_j, b_g_j, b_g_k, c_g_i, c_g_k):
        # A is 2x3. B is 3x2. C is 2x2.
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    a_idx = (a_g_i + i) * m + (a_g_j + j)
                    b_idx = (b_g_j + j) * p + (b_g_k + k)
                    c_idx = get_w_index(c_g_i + i, c_g_k + k)
                    add_term([a_idx], [b_idx], [(c_idx, 1)])

    # Top Half (i=0,1)
    # A_top1: i=0,1, j=0,1. (2x2)
    # B1: j=0,1, k=0..5. (2x6)
    # Split B1 into 3 blocks of 2x2: k=0,1; k=2,3; k=4,5.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2x2_global(0, 0, 0, 0, 0, 0)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2x2_global(0, 0, 0, 2, 0, 2)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_2x2x2_global(0, 0, 0, 4, 0, 4)
    
    # A_top2: i=0,1, j=2,3,4. (2x3)
    # B2: j=2,3,4, k=0..5. (3x6)
    # Split B2 into 3 blocks of 3x2: k=0,1; k=2,3; k=4,5.
    
    # Block 4: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    add_schoolbook_2x3x2_global(0, 2, 2, 0, 0, 0)
    # Block 5: A[0:2, 2:5] * B[2:5, 2:4] -> C[0:2, 2:4]
    add_schoolbook_2x3x2_global(0, 2, 2, 2, 0, 2)
    # Block 6: A[0:2, 2:5] * B[2:5, 4:6] -> C[0:2, 4:6]
    add_schoolbook_2x3x2_global(0, 2, 2, 4, 0, 4)
    
    # Bottom Half (i=2,3)
    # A_bot1: i=2,3, j=0,1. (2x2)
    # B1: j=0,1, k=0..5. (2x6)
    
    # Block 7: A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    add_strassen_2x2x2_global(2, 0, 0, 0, 2, 0)
    # Block 8: A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    add_strassen_2x2x2_global(2, 0, 0, 2, 2, 2)
    # Block 9: A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    add_strassen_2x2x2_global(2, 0, 0, 4, 2, 4)
    
    # A_bot2: i=2,3, j=2,3,4. (2x3)
    # B2: j=2,3,4, k=0..5. (3x6)
    
    # Block 10: A[2:4, 2:5] * B[2:5, 0:2] -> C[2:4, 0:2]
    add_schoolbook_2x3x2_global(2, 2, 2, 0, 2, 0)
    # Block 11: A[2:4, 2:5] * B[2:5, 2:4] -> C[2:4, 2:4]
    add_schoolbook_2x3x2_global(2, 2, 2, 2, 2, 2)
    # Block 12: A[2:4, 2:5] * B[2:5, 4:6] -> C[2:4, 4:6]
    add_schoolbook_2x3x2_global(2, 2, 2, 4, 2, 4)
    
    return {"U": U, "V": V, "W": W}
