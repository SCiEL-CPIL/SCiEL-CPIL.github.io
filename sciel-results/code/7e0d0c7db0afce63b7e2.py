import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        """
        Adds a term: coeff_u * A[i,k] * coeff_v * B[k,j] -> coeff_w * C[i,j]
        u_vec has coeff_u at i*m+k
        v_vec has coeff_v at k*p+j
        w_vec has coeff_w at j*n+i
        """
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        
        add_rank1(u_vec, v_vec, w_vec)
    
    # Part 1: 4x4x4 multiplication (k in 0..3, j in 0..3) using Strassen
    # Split 4x4 into 2x2 blocks:
    # A blocks: A00 (i=0,1; k=0,1), A01 (i=0,1; k=2,3), A10 (i=2,3; k=0,1), A11 (i=2,3; k=2,3)
    # B blocks: B00 (k=0,1; j=0,1), B01 (k=0,1; j=2,3), B10 (k=2,3; j=0,1), B11 (k=2,3; j=2,3)
    # C blocks: C00 (i=0,1; j=0,1), C01 (i=0,1; j=2,3), C10 (i=2,3; j=0,1), C11 (i=2,3; j=2,3)
    
    # Strassen 7 products for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen product, we need 4 schoolbook terms (2x2x2)
    # Each term: (i_local, k_local, j_local) within the 2x2 block
    
    # Define the 7 Strassen products as (A_form, B_form) where forms are lists of (block, sign)
    # A_form: list of (block_id, sign) for A blocks
    # B_form: list of (block_id, sign) for B blocks
    
    # Block index mapping for A (i,k):
    # A00: i in {0,1}, k in {0,1} -> global i, k
    # A01: i in {0,1}, k in {2,3} -> global i, k+2
    # A10: i in {2,3}, k in {0,1} -> global i+2, k
    # A11: i in {2,3}, k in {2,3} -> global i+2, k+2
    
    # Block index mapping for B (k,j):
    # B00: k in {0,1}, j in {0,1} -> global k, j
    # B01: k in {0,1}, j in {2,3} -> global k, j+2
    # B10: k in {2,3}, j in {0,1} -> global k+2, j
    # B11: k in {2,3}, j in {2,3} -> global k+2, j+2
    
    # Block index mapping for C (i,j):
    # C00: i in {0,1}, j in {0,1} -> global i, j
    # C01: i in {0,1}, j in {2,3} -> global i, j+2
    # C10: i in {2,3}, j in {0,1} -> global i+2, j
    # C11: i in {2,3}, j in {2,3} -> global i+2, j+2
    
    # Strassen products definition:
    # Each product has A_form and B_form as dicts: {block_id: sign}
    strassen_products = [
        # M1 = (A00+A11)(B00+B11)
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},
        # M2 = (A10+A11)B00
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},
        # M3 = A00(B01-B11)
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},
        # M4 = A11(B10-B00)
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},
        # M5 = (A00+A01)B11
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},
        # M6 = (A10-A00)(B00+B01)
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},
        # M7 = (A01-A11)(B10+B11)
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},
    ]
    
    # C block contributions for each Strassen product:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_contributions = [
        # For M1 (index 0): contributes to C00 (+1), C11 (+1)
        {0: 1, 3: 1},
        # For M2 (index 1): contributes to C10 (+1), C11 (-1)
        {2: 1, 3: -1},
        # For M3 (index 2): contributes to C01 (+1), C11 (+1)
        {1: 1, 3: 1},
        # For M4 (index 3): contributes to C00 (+1), C10 (+1)
        {0: 1, 2: 1},
        # For M5 (index 4): contributes to C00 (-1), C01 (+1)
        {0: -1, 1: 1},
        # For M6 (index 5): contributes to C11 (+1)
        {3: 1},
        # For M7 (index 6): contributes to C00 (+1)
        {0: 1},
    ]
    
    # Helper to get global indices from block and local indices
    def a_global(block, li, lk):
        # block 0: A00, block 1: A01, block 2: A10, block 3: A11
        if block == 0:
            return li, lk
        elif block == 1:
            return li, lk + 2
        elif block == 2:
            return li + 2, lk
        elif block == 3:
            return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        # block 0: B00, block 1: B01, block 2: B10, block 3: B11
        if block == 0:
            return lk, lj
        elif block == 1:
            return lk, lj + 2
        elif block == 2:
            return lk + 2, lj
        elif block == 3:
            return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        # block 0: C00, block 1: C01, block 2: C10, block 3: C11
        if block == 0:
            return li, lj
        elif block == 1:
            return li, lj + 2
        elif block == 2:
            return li + 2, lj
        elif block == 3:
            return li + 2, lj + 2
    
    # Inner Strassen for 2x2x2: 7 products, each with A/B/C terms as (local_i, local_k, sign) etc.
    # A terms: (li, lk, sign), B terms: (lk, lj, sign), C terms: (li, lj, sign)
    inner_strassen = [
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        # M2 = (A[1,0]+A[1,1])B[0,0]
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        # M3 = A[0,0](B[0,1]-B[1,1])
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        # M4 = A[1,1](B[1,0]-B[0,0])
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        # M5 = (A[0,0]+A[0,1])B[1,1]
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    # Generate 49 terms for the 4x4x4 part (7 outer Strassen x 7 inner Strassen)
    for prod_idx, prod in enumerate(strassen_products):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = c_contributions[prod_idx]
        
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            # A part: combine outer A_form with inner A terms
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            
            # B part: combine outer B_form with inner B terms
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            
            # C part: combine outer C_contrib with inner C terms
            for c_block, c_sign in c_contrib.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    
    # Part 3: 4x1x5 (k=4, j in 0..4) - schoolbook
    for i in range(4):
        for j in range(5):
            add_term(i, 4, j, 1.0, 1.0, 1.0)
    
    rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1
    
    U = np.stack(U_cols, axis=1)
    V = np.stack(V_cols, axis=1)
    W = np.stack(W_cols, axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
