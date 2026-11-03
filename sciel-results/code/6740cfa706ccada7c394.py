import numpy as np
import time

def construct():
    n, m, p = 2, 5, 6
    rank = 54
    
    # U: (n*m, rank) = (10, 54)
    # V: (m*p, rank) = (30, 54)
    # W: (p*n, rank) = (12, 54)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0 # current rank index
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal r
        if r >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx, r] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, r] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, r] += val
        r += 1

    # Helper to get index in flattened arrays
    def idx_A(i, j):
        return i * m + j
        
    def idx_B(j, k):
        return j * p + k
        
    def idx_C(k, i):
        return k * n + i

    # Strassen block for i in {0,1}, j in {j1, j2}, k in {k1, k2}
    def add_strassen_block(j1, j2, k1, k2):
        # A indices
        a00 = idx_A(0, j1)
        a01 = idx_A(0, j2)
        a10 = idx_A(1, j1)
        a11 = idx_A(1, j2)
        
        # B indices
        b00 = idx_B(j1, k1)
        b01 = idx_B(j1, k2)
        b10 = idx_B(j2, k1)
        b11 = idx_B(j2, k2)
        
        # C indices
        c00 = idx_C(k1, 0)
        c01 = idx_C(k2, 0)
        c10 = idx_C(k1, 1)
        c11 = idx_C(k2, 1)
        
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += M1, C11 += M1
        add_term([a00, a11], [b00, b11], [c00, c11], [1, 1], [1, 1], [1, 1])
        
        # M2 = (A10 + A11)B00
        # C10 += M2, C11 -= M2
        add_term([a10, a11], [b00], [c10, c11], [1, 1], [1], [1, -1])
        
        # M3 = A00(B01 - B11)
        # C01 += M3, C11 += M3
        add_term([a00], [b01, b11], [c01, c11], [1], [1, -1], [1, 1])
        
        # M4 = A11(B10 - B00)
        # C00 += M4, C10 += M4
        add_term([a11], [b10, b00], [c00, c10], [1], [1, -1], [1, 1])
        
        # M5 = (A00 + A01)B11
        # C00 -= M5, C01 += M5
        add_term([a00, a01], [b11], [c00, c01], [1, 1], [1], [-1, 1])
        
        # M6 = (A10 - A00)(B00 + B01)
        # C11 += M6
        add_term([a10, a00], [b00, b01], [c11], [1, -1], [1, 1], [1])
        
        # M7 = (A01 - A11)(B10 + B11)
        # C00 += M7
        add_term([a01, a11], [b10, b11], [c00], [1, -1], [1, 1], [1])

    # Schoolbook block for i in {0,1}, j in {j1}, k in {k1, k2}
    def add_schoolbook_1x2(j1, k1, k2):
        # C0k1 += A0j1 B j1k1
        # C0k2 += A0j1 B j1k2
        # C1k1 += A1j1 B j1k1
        # C1k2 += A1j1 B j1k2
        
        a0 = idx_A(0, j1)
        a1 = idx_A(1, j1)
        
        b1 = idx_B(j1, k1)
        b2 = idx_B(j1, k2)
        
        c01 = idx_C(k1, 0)
        c02 = idx_C(k2, 0)
        c11 = idx_C(k1, 1)
        c12 = idx_C(k2, 1)
        
        # Term 1: A0j1 * B j1k1 -> C0k1
        add_term([a0], [b1], [c01], [1], [1], [1])
        
        # Term 2: A0j1 * B j1k2 -> C0k2
        add_term([a0], [b2], [c02], [1], [1], [1])
        
        # Term 3: A1j1 * B j1k1 -> C1k1
        add_term([a1], [b1], [c11], [1], [1], [1])
        
        # Term 4: A1j1 * B j1k2 -> C1k2
        add_term([a1], [b2], [c12], [1], [1], [1])

    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4, 5)]

    for jg in j_groups:
        for kg in k_groups:
            if len(jg) == 2 and len(kg) == 2:
                add_strassen_block(jg[0], jg[1], kg[0], kg[1])
            elif len(jg) == 1 and len(kg) == 2:
                add_schoolbook_1x2(jg[0], kg[0], kg[1])
            # Other cases not present in this grouping

    return {'U': U, 'V': V, 'W': W}
