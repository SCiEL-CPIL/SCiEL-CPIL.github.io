import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 5, 5
    
    U = []
    V = []
    W = []
    
    def add_tensor(u_idx, v_idx, w_idx, coeff=1):
        # u_idx, v_idx, w_idx are tuples of indices for U, V, W
        # We need to add a rank-1 tensor: U[:, r] = u, V[:, r] = v, W[:, r] = w
        # But we are building the matrices column by column (rank dimension).
        # So we append a new column to U, V, W.
        col_u = [0] * (n * m)
        col_v = [0] * (m * p)
        col_w = [0] * (p * n)
        
        for idx, val in zip(u_idx, [coeff] * len(u_idx)):
            col_u[idx] = val
        for idx, val in zip(v_idx, [coeff] * len(v_idx)):
            col_v[idx] = val
        for idx, val in zip(w_idx, [coeff] * len(w_idx)):
            col_w[idx] = val
            
        U.append(col_u)
        V.append(col_v)
        W.append(col_w)

    # Helper to add a block multiplication
    # A block: rows i1..i2, cols j1..j2
    # B block: rows j1..j2, cols k1..k2
    # C block: rows i1..i2, cols k1..k2
    # This assumes the block multiplication is done via schoolbook or Strassen.
    
    # We will use a recursive function to decompose a multiplication of size n x m x p
    # into smaller blocks, using Strassen for 2x2x2.
    
    def decompose(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        # i_off, j_off, k_off are offsets in the original matrices
        # We add rank-1 tensors for this sub-problem
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Use Strassen
            # A indices: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
            # B indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
            # C indices: (i_off, k_off), (i_off, k_off+1), (i_off+1, k_off), (i_off+1, k_off+1)
            
            a00 = i_off * m + j_off
            a01 = i_off * m + j_off + 1
            a10 = (i_off + 1) * m + j_off
            a11 = (i_off + 1) * m + j_off + 1
            
            b00 = j_off * p + k_off
            b01 = j_off * p + k_off + 1
            b10 = (j_off + 1) * p + k_off
            b11 = (j_off + 1) * p + k_off + 1
            
            c00 = k_off * n + i_off
            c01 = k_off * n + i_off + 1
            c10 = (k_off + 1) * n + i_off
            c11 = (k_off + 1) * n + i_off + 1
            
            # Strassen products
            # M1 = (A00+A11)(B00+B11)
            add_tensor([a00, a11], [b00, b11], [c00, c01, c10, c11]) # Coeffs need to be specific
            # Wait, add_tensor adds coeff to all. Strassen has different coeffs.
            # I need a more flexible add_tensor.
            
            # Let's redefine add_tensor to take lists of (index, coeff)
            
    # Redefining add_tensor for flexibility
    U.clear(); V.clear(); W.clear()
    
    def add_tensor(u_pairs, v_pairs, w_pairs):
        col_u = [0] * (n * m)
        col_v = [0] * (m * p)
        col_w = [0] * (p * n)
        
        for idx, val in u_pairs:
            col_u[idx] = val
        for idx, val in v_pairs:
            col_v[idx] = val
        for idx, val in w_pairs:
            col_w[idx] = val
            
        U.append(col_u)
        V.append(col_v)
        W.append(col_w)

    def strassen_2x2x2(i_off, j_off, k_off):
        a00 = i_off * m + j_off
        a01 = i_off * m + j_off + 1
        a10 = (i_off + 1) * m + j_off
        a11 = (i_off + 1) * m + j_off + 1
        
        b00 = j_off * p + k_off
        b01 = j_off * p + k_off + 1
        b10 = (j_off + 1) * p + k_off
        b11 = (j_off + 1) * p + k_off + 1
        
        c00 = k_off * n + i_off
        c01 = k_off * n + i_off + 1
        c10 = (k_off + 1) * n + i_off
        c11 = (k_off + 1) * n + i_off + 1
        
        # M1 = (A00+A11)(B00+B11)
        add_tensor([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c01, 1), (c10, 1), (c11, -1)])
        
        # M2 = (A10+A11)B00
        add_tensor([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, 1)])
        
        # M3 = A00(B10-B00)
        add_tensor([(a00, 1)], [(b10, 1), (b00, -1)], [(c10, 1), (c11, 1)])
        
        # M4 = A11(B01-B10)
        add_tensor([(a11, 1)], [(b01, 1), (b10, -1)], [(c01, 1), (c11, 1)])
        
        # M5 = (A00+A01)B11
        add_tensor([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, 1), (c01, 1)])
        
        # M6 = (A10-A00)(B00+B01)
        add_tensor([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c10, 1), (c00, -1)])
        
        # M7 = (A11-A01)(B01+B11)
        add_tensor([(a11, 1), (a01, -1)], [(b01, 1), (b11, 1)], [(c11, 1), (c01, -1)])

    def schoolbook_1x1x1(i_off, j_off, k_off):
        a_idx = i_off * m + j_off
        b_idx = j_off * p + k_off
        c_idx = k_off * n + i_off
        add_tensor([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])

    # Decomposition Strategy:
    # 1. Split A (2x5) into A1 (2x4) and A2 (2x1)
    # 2. Split B (5x5) into B1 (4x5) and B2 (1x5)
    # C = A1 B1 + A2 B2
    
    # Part 2: A2 (2x1) * B2 (1x5) -> Rank 5
    for k in range(5):
        schoolbook_1x1x1(0, 4, k) # A2 is col 4, B2 is row 4
        schoolbook_1x1x1(1, 4, k)
        
    # Part 1: A1 (2x4) * B1 (4x5)
    # Split B1 (4x5) into B11 (4x4) and B12 (4x1)
    # A1 B1 = A1 B11 + A1 B12
    
    # Part 1b: A1 (2x4) * B12 (4x1) -> Rank 8
    for k in range(4): # k is 0..3 in B1, which is col 4 in B? No, B1 is cols 0..3 of B?
        # B1 is rows 0..3 of B. B12 is col 4 of B1? No.
        # B1 is 4x5. B11 is 4x4 (cols 0..3), B12 is 4x1 (col 4).
        # So B12 corresponds to B rows 0..3, col 4.
        for i in range(2):
            for j in range(4):
                a_idx = i * m + j
                b_idx = j * p + 4
                c_idx = 4 * n + i
                add_tensor([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
                
    # Part 1a: A1 (2x4) * B11 (4x4)
    # Split A1 (2x4) into A11 (2x2) and A12 (2x2)
    # Split B11 (4x4) into B111 (2x4) and B112 (2x4)
    # A1 B11 = A11 B111 + A12 B112
    
    # Part 1a-i: A11 (2x2) * B111 (2x4)
    # B111 is rows 0..1, cols 0..3 of B.
    # Split B111 into two 2x2 blocks: B111a (cols 0..1) and B111b (cols 2..3)
    # A11 B111 = A11 B111a + A11 B111b
    # Each is 2x2x2. Use Strassen.
    
    # A11 is A rows 0..1, cols 0..1.
    # B111a is B rows 0..1, cols 0..1.
    strassen_2x2x2(0, 0, 0)
    
    # B111b is B rows 0..1, cols 2..3.
    strassen_2x2x2(0, 0, 2)
    
    # Part 1a-ii: A12 (2x2) * B112 (2x4)
    # A12 is A rows 0..1, cols 2..3.
    # B112 is B rows 2..3, cols 0..3.
    # Split B112 into B112a (cols 0..1) and B112b (cols 2..3)
    
    # B112a is B rows 2..3, cols 0..1.
    strassen_2x2x2(0, 2, 0)
    
    # B112b is B rows 2..3, cols 2..3.
    strassen_2x2x2(0, 2, 2)
    
    # Total rank: 5 (A2B2) + 8 (A1B12) + 7 (A11B111a) + 7 (A11B111b) + 7 (A12B112a) + 7 (A12B112b) = 41
    
    return {"U": U, "V": V, "W": W}
