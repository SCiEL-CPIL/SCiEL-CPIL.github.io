def construct():
    n, m, p = 2, 4, 4
    U = []
    V = []
    W = []
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # Strassen 7 multiplications
        # Local indices 0,1
        # Global indices mapped via lists
        
        def u_idx(li, lj): return a_rows[li] * m + a_cols[lj]
        def v_idx(lj, lk): return b_rows[lj] * p + b_cols[lk]
        def w_idx(lk, li): return c_cols[lk] * n + c_rows[li]
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        u1 = {u_idx(0,0): 1, u_idx(1,1): 1}
        v1 = {v_idx(0,0): 1, v_idx(1,1): 1}
        w1 = {w_idx(0,0): 1, w_idx(1,1): 1}
        
        # M2: (A10+A11)B00 -> C10, C11
        u2 = {u_idx(1,0): 1, u_idx(1,1): 1}
        v2 = {v_idx(0,0): 1}
        w2 = {w_idx(0,1): 1, w_idx(1,1): -1} # C10 = M2+M4, C11 = M1-M2+...
        # Wait, C10 index is w_idx(0,1)?
        # C[i,k]. i=1, k=0.
        # w_idx(lk, li) -> c_cols[lk] * n + c_rows[li].
        # C10: i=1, k=0.
        # In local coords: i=1, k=0.
        # w_idx(0, 1) -> c_cols[0]*n + c_rows[1]. Correct.
        
        # M3: A00(B10-B11) -> C01, C11
        u3 = {u_idx(0,0): 1}
        v3 = {v_idx(1,0): 1, v_idx(1,1): -1}
        w3 = {w_idx(1,0): 1, w_idx(1,1): 1} # C01 = M3+M5, C11 = ...+M3
        # C01: i=0, k=1. w_idx(1,0).
        # C11: i=1, k=1. w_idx(1,1).
        
        # M4: A11(B01-B00) -> C00, C10
        u4 = {u_idx(1,1): 1}
        v4 = {v_idx(0,1): 1, v_idx(0,0): -1}
        w4 = {w_idx(0,0): 1, w_idx(0,1): 1} # C00 = M1+M4..., C10 = M2+M4
        # C00: i=0, k=0. w_idx(0,0).
        # C10: i=1, k=0. w_idx(0,1).
        
        # M5: (A00+A10)B11 -> C00, C01
        u5 = {u_idx(0,0): 1, u_idx(1,0): 1}
        v5 = {v_idx(1,1): 1}
        w5 = {w_idx(0,0): -1, w_idx(1,0): 1} # C00 = ...-M5, C01 = M3+M5
        # C00: i=0, k=0. w_idx(0,0).
        # C01: i=0, k=1. w_idx(1,0).
        
        # M6: (A10-A00)(B00+B10) -> C11
        u6 = {u_idx(1,0): 1, u_idx(0,0): -1}
        v6 = {v_idx(0,0): 1, v_idx(1,0): 1}
        w6 = {w_idx(1,1): 1} # C11 = ...+M6
        # C11: i=1, k=1. w_idx(1,1).
        
        # M7: (A01-A11)(B11+B01) -> C00
        u7 = {u_idx(0,1): 1, u_idx(1,1): -1}
        v7 = {v_idx(1,1): 1, v_idx(0,1): 1}
        w7 = {w_idx(0,0): 1} # C00 = ...+M7
        # C00: i=0, k=0. w_idx(0,0).
        
        blocks = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for u_d, v_d, w_d in blocks:
            # Append to U, V, W
            # U is list of lists. We need to add a column.
            # If U is empty, initialize.
            if not U:
                U = [[0] for _ in range(n*m)]
                V = [[0] for _ in range(m*p)]
                W = [[0] for _ in range(p*n)]
                
            # Get max index to ensure size
            max_u = max(u_d.keys())
            max_v = max(v_d.keys())
            max_w = max(w_d.keys())
            
            # Resize if necessary (should not be needed if order is correct)
            # But safe to check
            
            u_col = [0] * (n*m)
            for idx, val in u_d.items():
                u_col[idx] = val
            U.append(u_col)
            
            v_col = [0] * (m*p)
            for idx, val in v_d.items():
                v_col[idx] = val
            V.append(v_col)
            
            w_col = [0] * (p*n)
            for idx, val in w_d.items():
                w_col[idx] = val
            W.append(w_col)

    # Call for 4 blocks
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])

    return {"U": U, "V": V, "W": W}
