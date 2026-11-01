"""Rewritten program with Strassen-based block decomposition for rank minimization."""
DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # Rank = 7 * 7 (Strassen for 2x2x2) + 4 (Schoolbook for 2x2x1) = 53
    R = 53
    
    U = [[0] * R for _ in range(n * m)]  # 4 x 53
    V = [[0] * R for _ in range(m * p)]  # 30 x 53
    W = [[0] * R for _ in range(p * n)]  # 30 x 53
    
    def add_term(U_row, V_row, W_row, col, val=1):
        if isinstance(U_row, list):
            for r, v in zip(U_row, val):
                if v != 0:
                    U[r][col] += v
        else:
            U[U_row][col] += val
        if isinstance(V_row, list):
            for r, v in zip(V_row, val):
                if v != 0:
                    V[r][col] += v
        else:
            V[V_row][col] += val
        if isinstance(W_row, list):
            for r, v in zip(W_row, val):
                if v != 0:
                    W[r][col] += v
        else:
            W[W_row][col] += val

    col = 0
    # 7 Strassen blocks for k=0..13
    for b in range(7):
        # Global column offset for B and C
        k_offset = 2 * b
        
        # Strassen Terms for 2x2x2
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # U: (0,0)=1, (1,1)=1. V: (0,0)=1, (1,1)=1. W: C00, C11
        add_term([0, 3], [k_offset, k_offset+1], [k_offset*2, k_offset*2+1], col)
        col += 1
        
        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        # U: (1,0)=2, (1,1)=3. V: (0,0)=k_offset. W: C10, C11(-1)
        add_term([2, 3], [k_offset], [k_offset*2+1, -(k_offset*2+1)], col)
        col += 1
        
        # M3 = A00(B10-B11) -> C01 += 1, C11 -= 1
        # U: (0,0)=0. V: (1,0)=k_offset+1, (1,1)=k_offset+2. W: C01, C11(-1)
        add_term([0], [k_offset+1, k_offset+2], [k_offset*2+1, -(k_offset*2+1)], col)
        # Wait, M3 = A00(B10-B11). B10 is row 1 col 0. B11 is row 1 col 1.
        # V index for B10 is 1*p + k_offset. V index for B11 is 1*p + k_offset+1.
        # p=15? No, V is (m*p, R) = (2*15, R). Index j*p + k.
        # B10: j=1, k=k_offset -> 1*15 + k_offset.
        # B11: j=1, k=k_offset+1 -> 1*15 + k_offset+1.
        # I made a mistake in V index above.
        # Let's restart V index calculation properly.
        pass
    
    # Reset and do it carefully
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    
    # Helper to set values
    def set_term(u_vals, v_vals, w_vals, c):
        for i, val in u_vals.items():
            U[i][c] += val
        for i, val in v_vals.items():
            V[i][c] += val
        for i, val in w_vals.items():
            W[i][c] += val

    # Strassen Blocks
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices: B_jk is at j*p + k
        v_b00 = 0 * p + k0
        v_b01 = 0 * p + k1
        v_b10 = 1 * p + k0
        v_b11 = 1 * p + k1
        
        # W indices: C_ki is at k*n + i
        w_c00 = k0 * n + 0
        w_c01 = k0 * n + 1
        w_c10 = k1 * n + 0
        w_c11 = k1 * n + 1
        
        # M1: (A00+A11)(B00+B11)
        # U: A00(0), A11(3)
        # V: B00, B11
        # W: C00(+1), C11(+1)
        set_term({0:1, 3:1}, {v_b00:1, v_b11:1}, {w_c00:1, w_c11:1}, col)
        col += 1
        
        # M2: (A10+A11)B00
        # U: A10(2), A11(3)
        # V: B00
        # W: C10(+1), C11(-1)
        set_term({2:1, 3:1}, {v_b00:1}, {w_c10:1, w_c11:-1}, col)
        col += 1
        
        # M3: A00(B10-B11)
        # U: A00(0)
        # V: B10(+1), B11(-1)
        # W: C01(+1), C11(-1)
        set_term({0:1}, {v_b10:1, v_b11:-1}, {w_c01:1, w_c11:-1}, col)
        col += 1
        
        # M4: A11(B01+B10)
        # U: A11(3)
        # V: B01(+1), B10(+1)
        # W: C01(+1), C10(+1)
        set_term({3:1}, {v_b01:1, v_b10:1}, {w_c01:1, w_c10:1}, col)
        col += 1
        
        # M5: (A00+A10)B11
        # U: A00(0), A10(2)
        # V: B11
        # W: C00(+1), C10(+1)
        set_term({0:1, 2:1}, {v_b11:1}, {w_c00:1, w_c10:1}, col)
        col += 1
        
        # M6: (A11-A00)(B00+B01)
        # U: A11(1), A00(-1)
        # V: B00(+1), B01(+1)
        # W: C10(+1), C11(-1)
        set_term({3:1, 0:-1}, {v_b00:1, v_b01:1}, {w_c10:1, w_c11:-1}, col)
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        # U: A01(1), A11(-1)
        # V: B10(+1), B11(+1)
        # W: C00(+1), C11(+1)
        set_term({1:1, 3:-1}, {v_b10:1, v_b11:1}, {w_c00:1, w_c11:1}, col)
        col += 1

    # Schoolbook Block for k=14
    k14 = 14
    v_b0_14 = 0 * p + k14
    v_b1_14 = 1 * p + k14
    
    w_c0_14 = k14 * n + 0
    w_c1_14 = k14 * n + 1
    
    # T1: A00 B0_14 -> C0_14
    set_term({0:1}, {v_b0_14:1}, {w_c0_14:1}, col)
    col += 1
    
    # T2: A01 B1_14 -> C0_14
    set_term({1:1}, {v_b1_14:1}, {w_c0_14:1}, col)
    col += 1
    
    # T3: A10 B0_14 -> C1_14
    set_term({2:1}, {v_b0_14:1}, {w_c1_14:1}, col)
    col += 1
    
    # T4: A11 B1_14 -> C1_14
    set_term({3:1}, {v_b1_14:1}, {w_c1_14:1}, col)
    col += 1
    
    return {"U": U, "V": V, "W": W}
