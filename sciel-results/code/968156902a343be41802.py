import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 0
    U = []
    V = []
    W = []

    def add_term(u_val, v_val, w_val):
        nonlocal rank
        # u_val is dict { (i,j): coeff }
        # v_val is dict { (j,k): coeff }
        # w_val is dict { (k,i): coeff }
        for _ in range(1):
            u_row = [0.0] * (n*m)
            v_row = [0.0] * (m*p)
            w_row = [0.0] * (p*n)
            
            for (i, j), c in u_val.items():
                u_row[i*m + j] += c
            for (j, k), c in v_val.items():
                v_row[j*p + k] += c
            for (k, i), c in w_val.items():
                w_row[k*n + i] += c
            
            U.append(u_row)
            V.append(v_row)
            W.append(w_row)
            rank += 1

    # Helper for Strassen 2x2 * 2x2
    # A indices (i, j_offset), B indices (j_offset, k_offset)
    # Returns list of (u, v, w) terms
    def strassen_2x2(i_off, j_off, k_off):
        # A is 2x2 at A[i_off:i_off+2, j_off:j_off+2]
        # B is 2x2 at B[j_off:j_off+2, k_off:k_off+2]
        # C is 2x2 at C[i_off:i_off+2, k_off:k_off+2]
        
        # M1 = (a00+a11)(b00+b11)
        u1 = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        v1 = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        w1 = {(k_off, i_off): 1, (k_off, i_off+1): 1, (k_off+1, i_off): 1, (k_off+1, i_off+1): 1}
        add_term(u1, v1, w1)
        
        # M2 = (a10+a11)b00
        u2 = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        v2 = {(j_off, k_off): 1}
        w2 = {(k_off, i_off+1): 1}
        add_term(u2, v2, w2)
        
        # M3 = a00(b01-b11)
        u3 = {(i_off, j_off): 1}
        v3 = {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1}
        w3 = {(k_off+1, i_off): 1}
        add_term(u3, v3, w3)
        
        # M4 = a11(b10-b00)
        u4 = {(i_off+1, j_off+1): 1}
        v4 = {(j_off+1, k_off): 1, (j_off, k_off): -1}
        w4 = {(k_off, i_off+1): 1}
        add_term(u4, v4, w4)
        
        # M5 = (a00+a01)b11
        u5 = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        v5 = {(j_off+1, k_off+1): 1}
        w5 = {(k_off+1, i_off): 1}
        add_term(u5, v5, w5)
        
        # M6 = (a10-a00)(b00+b01)
        u6 = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        v6 = {(j_off, k_off): 1, (j_off, k_off+1): 1}
        w6 = {(k_off, i_off+1): 1, (k_off+1, i_off+1): 1}
        add_term(u6, v6, w6)
        
        # M7 = (a01-a11)(b10+b11)
        u7 = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        v7 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
        w7 = {(k_off, i_off): 1, (k_off+1, i_off): 1}
        add_term(u7, v7, w7)

    # Helper for Schoolbook 2x2 * 2x1
    def schoolbook_2x2_1(i_off, j_off, k_off):
        # A 2x2, B 2x1, C 2x1
        # C00 = a00 b00 + a01 b10
        # C10 = a10 b00 + a11 b10
        
        # Term 1: a00 b00
        u1 = {(i_off, j_off): 1}
        v1 = {(j_off, k_off): 1}
        w1 = {(k_off, i_off): 1}
        add_term(u1, v1, w1)
        
        # Term 2: a01 b10
        u2 = {(i_off, j_off+1): 1}
        v2 = {(j_off+1, k_off): 1}
        w2 = {(k_off, i_off): 1}
        add_term(u2, v2, w2)
        
        # Term 3: a10 b00
        u3 = {(i_off+1, j_off): 1}
        v3 = {(j_off, k_off): 1}
        w3 = {(k_off, i_off+1): 1}
        add_term(u3, v3, w3)
        
        # Term 4: a11 b10
        u4 = {(i_off+1, j_off+1): 1}
        v4 = {(j_off+1, k_off): 1}
        w4 = {(k_off, i_off+1): 1}
        add_term(u4, v4, w4)

    # Block 1: A[:, 0:2] * B[0:2, :]
    # B[0:2, :] is 2x5. Split into 2x2, 2x2, 2x1.
    strassen_2x2(0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2]
    strassen_2x2(0, 0, 2) # A[0:2, 0:2] * B[0:2, 2:4]
    schoolbook_2x2_1(0, 0, 4) # A[0:2, 0:2] * B[0:2, 4:5]
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    # B[2:4, :] is 2x5. Split into 2x2, 2x2, 2x1.
    strassen_2x2(0, 2, 0) # A[0:2, 2:4] * B[2:4, 0:2]
    strassen_2x2(0, 2, 2) # A[0:2, 2:4] * B[2:4, 2:4]
    schoolbook_2x2_1(0, 2, 4) # A[0:2, 2:4] * B[2:4, 4:5]

    return {"U": U, "V": V, "W": W}
