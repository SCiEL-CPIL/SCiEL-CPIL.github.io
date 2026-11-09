import time

_START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            if c != 0:
                U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            if c != 0:
                V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            if c != 0:
                W[idx][col] += c

    current_col = 0
    
    # We decompose the 2x3x8 multiplication into 4 independent 2x3x2 multiplications
    # based on the output columns k.
    # Each 2x3x2 multiplication is decomposed into:
    # - 2x2x2 for j in {0, 1} (Strassen, rank 7)
    # - 2x1x2 for j = 2 (Outer product, rank 4)
    
    for k_block in range(4):
        if time.time() - _START_TIME > TIME_LIMIT:
            break
            
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        # Indices for A (2x3): 0,1,2,3,4,5
        # Row 0: 0, 1, 2
        # Row 1: 3, 4, 5
        a_idx = [0, 1, 2]
        b_idx = [3, 4, 5]
        
        # Indices for B (3x2 block):
        # Row 0: 0*p+k0, 0*p+k1
        # Row 1: 1*p+k0, 1*p+k1
        # Row 2: 2*p+k0, 2*p+k1
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        g_idx = 1 * p + k0
        h_idx = 1 * p + k1
        e2_idx = 2 * p + k0
        f2_idx = 2 * p + k1
        
        # Indices for C (2x2 block):
        # Row 0: k0*n, k1*n
        # Row 1: k0*n+1, k1*n+1
        w_00 = k0 * n
        w_01 = k1 * n
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # Strassen 2x2x2 for j=0,1
        # A_sub: rows 0,1; cols 0,1 -> indices 0,1,3,4
        # B_sub: rows 0,1; cols 0,1 -> indices e,f,g,h
        # C_sub: rows 0,1; cols 0,1 -> indices w_00, w_01, w_10, w_11
        
        # Standard Strassen for 2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Mapping to our indices:
        # a00=0, a01=1, a10=3, a11=4
        # b00=e, b01=f, b10=g, b11=h
        
        add_term(current_col, [0, 4], [1, 1], [e, h], [1, 1], [w_00, w_11], [1, 1]) # m1
        current_col += 1
        add_term(current_col, [3, 4], [1, 1], [e], [1], [w_10, w_11], [1, -1]) # m2
        current_col += 1
        add_term(current_col, [0], [1], [f, h], [1, -1], [w_01, w_11], [1, 1]) # m3
        current_col += 1
        add_term(current_col, [4], [1], [g, e], [1, -1], [w_00, w_10], [1, 1]) # m4
        current_col += 1
        add_term(current_col, [0, 1], [1, 1], [h], [1], [w_00, w_01], [-1, 1]) # m5
        current_col += 1
        add_term(current_col, [3, 0], [1, -1], [e, f], [1, 1], [w_11], [1]) # m6
        current_col += 1
        add_term(current_col, [1, 4], [1, -1], [g, h], [1, 1], [w_00], [1]) # m7
        current_col += 1
        
        # Outer product 2x1x2 for j=2
        # A_sub: col 2 -> indices 2, 5
        # B_sub: row 2 -> indices e2, f2
        # C_sub: all 4 indices
        
        add_term(current_col, [2], [1], [e2], [1], [w_00], [1])
        current_col += 1
        add_term(current_col, [2], [1], [f2], [1], [w_01], [1])
        current_col += 1
        add_term(current_col, [5], [1], [e2], [1], [w_10], [1])
        current_col += 1
        add_term(current_col, [5], [1], [f2], [1], [w_11], [1])
        current_col += 1

    return {"U": U, "V": V, "W": W}
