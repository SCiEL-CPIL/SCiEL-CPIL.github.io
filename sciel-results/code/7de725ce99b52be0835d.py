import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=5, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term_dict(u_dict, v_dict, w_dict):
        U_cols.append(u_dict)
        V_cols.append(v_dict)
        W_cols.append(w_dict)
        
    def idx_a(r, c):
        return r * m + c
        
    def idx_b(r, c):
        return r * p + c
        
    def idx_c(r, c):
        # W index convention: k*n + i
        # Here r is row i, c is col k
        return c * n + r

    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # Indices within the 2x2 blocks
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = {idx_a(a_r0+0, a_c0+0): 1, idx_a(a_r0+1, a_c0+1): 1}
        v1 = {idx_b(b_r0+0, b_c0+0): 1, idx_b(b_r0+1, b_c0+1): 1}
        w1 = {idx_c(c_r0+0, c_c0+0): 1, idx_c(c_r0+1, c_c0+1): 1}
        add_term_dict(u1, v1, w1)
        
        # M2 = (A10 + A11)B00
        u2 = {idx_a(a_r0+1, a_c0+0): 1, idx_a(a_r0+1, a_c0+1): 1}
        v2 = {idx_b(b_r0+0, b_c0+0): 1}
        w2 = {idx_c(c_r0+1, c_c0+0): 1, idx_c(c_r0+1, c_c0+1): -1}
        add_term_dict(u2, v2, w2)
        
        # M3 = A00(B10 - B11)
        u3 = {idx_a(a_r0+0, a_c0+0): 1}
        v3 = {idx_b(b_r0+1, b_c0+0): 1, idx_b(b_r0+1, b_c0+1): -1}
        w3 = {idx_c(c_r0+0, c_c0+1): 1, idx_c(c_r0+1, c_c0+1): 1}
        add_term_dict(u3, v3, w3)
        
        # M4 = A11(B01 - B00)
        u4 = {idx_a(a_r0+1, a_c0+1): 1}
        v4 = {idx_b(b_r0+0, b_c0+1): 1, idx_b(b_r0+0, b_c0+0): -1}
        w4 = {idx_c(c_r0+0, c_c0+0): 1, idx_c(c_r0+1, c_c0+0): 1}
        add_term_dict(u4, v4, w4)
        
        # M5 = (A00 + A01)B11
        u5 = {idx_a(a_r0+0, a_c0+0): 1, idx_a(a_r0+0, a_c0+1): 1}
        v5 = {idx_b(b_r0+1, b_c0+1): 1}
        w5 = {idx_c(c_r0+0, c_c0+0): -1, idx_c(c_r0+0, c_c0+1): 1}
        add_term_dict(u5, v5, w5)
        
        # M6 = (A10 - A00)(B00 + B01)
        u6 = {idx_a(a_r0+1, a_c0+0): 1, idx_a(a_r0+0, a_c0+0): -1}
        v6 = {idx_b(b_r0+0, b_c0+0): 1, idx_b(b_r0+0, b_c0+1): 1}
        w6 = {idx_c(c_r0+1, c_c0+1): 1}
        add_term_dict(u6, v6, w6)
        
        # M7 = (A01 - A11)(B10 + B11)
        u7 = {idx_a(a_r0+0, a_c0+1): 1, idx_a(a_r0+1, a_c0+1): -1}
        v7 = {idx_b(b_r0+1, b_c0+0): 1, idx_b(b_r0+1, b_c0+1): 1}
        w7 = {idx_c(c_r0+0, c_c0+0): 1}
        add_term_dict(u7, v7, w7)

    def add_schoolbook(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, a_h, a_w, b_h, b_w, c_h, c_w):
        # a_h, a_w: height, width of A block
        # b_h, b_w: height, width of B block
        # c_h, c_w: height, width of C block
        # assert a_w == b_h
        for i in range(a_h):
            for k in range(b_w):
                for j in range(a_w):
                    u = {idx_a(a_r0+i, a_c0+j): 1}
                    v = {idx_b(b_r0+j, b_c0+k): 1}
                    w = {idx_c(c_r0+i, c_c0+k): 1}
                    add_term_dict(u, v, w)

    # 1. C11 (2x2) = A11 (2x2) * B11 (2x2) -> Strassen
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # 2. C22 (2x2) = A22 (2x2) * B22 (2x2) -> Strassen
    add_strassen_2x2(2, 2, 2, 2, 2, 2)
    
    # 3. C33 (1x1) = A33 (1x1) * B33 (1x1) -> Schoolbook
    add_schoolbook(4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1)
    
    # 4. C12 (2x2) = A11 (2x2) * B12 (2x2) + A12 (2x2) * B22 (2x2)
    # Part 1
    add_schoolbook(0, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2)
    # Part 2
    add_schoolbook(0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2)
    
    # 5. C21 (2x2) = A21 (2x2) * B11 (2x2) + A22 (2x2) * B21 (2x2)
    # Part 1
    add_schoolbook(2, 0, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2)
    # Part 2
    add_schoolbook(2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2)
    
    # 6. C13 (2x1) = A11 (2x2) * B13 (2x1) + A12 (2x2) * B23 (2x1) + A13 (2x1) * B33 (1x1)
    add_schoolbook(0, 0, 0, 4, 0, 4, 2, 2, 2, 1, 2, 1)
    add_schoolbook(0, 2, 2, 4, 0, 4, 2, 2, 2, 1, 2, 1)
    add_schoolbook(0, 4, 4, 4, 0, 4, 2, 1, 1, 1, 2, 1)
    
    # 7. C23 (2x1) = A21 (2x2) * B13 (2x1) + A22 (2x2) * B23 (2x1) + A23 (2x1) * B33 (1x1)
    add_schoolbook(2, 0, 0, 4, 2, 4, 2, 2, 2, 1, 2, 1)
    add_schoolbook(2, 2, 2, 4, 2, 4, 2, 2, 2, 1, 2, 1)
    add_schoolbook(2, 4, 4, 4, 2, 4, 2, 1, 1, 1, 2, 1)
    
    # 8. C31 (1x2) = A31 (1x2) * B11 (2x2) + A32 (1x2) * B21 (2x2) + A33 (1x1) * B31 (1x2)
    add_schoolbook(4, 0, 0, 0, 4, 0, 1, 2, 2, 2, 1, 2)
    add_schoolbook(4, 2, 2, 0, 4, 0, 1, 2, 2, 2, 1, 2)
    add_schoolbook(4, 4, 4, 0, 4, 0, 1, 1, 1, 2, 1, 2)
    
    # 9. C32 (1x2) = A31 (1x2) * B12 (2x2) + A32 (1x2) * B22 (2x2) + A33 (1x1) * B32 (1x2)
    add_schoolbook(4, 0, 0, 2, 4, 2, 1, 2, 2, 2, 1, 2)
    add_schoolbook(4, 2, 2, 2, 4, 2, 1, 2, 2, 2, 1, 2)
    add_schoolbook(4, 4, 4, 2, 4, 2, 1, 1, 1, 2, 1, 2)

    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
