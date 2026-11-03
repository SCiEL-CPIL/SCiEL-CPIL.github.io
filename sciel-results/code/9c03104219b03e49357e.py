DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for 2x5x6 (rank 49) and 1x5x6 (rank 30)
    # Total rank 79
    
    rank = 79
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a naive 2x2x2 block
    def add_naive_2x2x2(row_offset, mid_offset, out_offset):
        nonlocal col_idx
        # row_offset: 0 or 1 (relative to the 2 rows)
        # mid_offset: start index in mid dim
        # out_offset: start index in out dim
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = (row_offset + i) * m + (mid_offset + j)
                    v_idx = (mid_offset + j) * p + (out_offset + k)
                    w_idx = (out_offset + k) * n + (row_offset + i)
                    
                    U[u_idx][col_idx] = 1
                    V[v_idx][col_idx] = 1
                    W[w_idx][col_idx] = 1
                    col_idx += 1
                    if col_idx > rank: raise ValueError("Rank exceeded")

    # Helper to add a naive 2x2x1 block
    def add_naive_2x2x1(row_offset, mid_offset, out_offset):
        nonlocal col_idx
        for i in range(2):
            for j in range(2):
                u_idx = (row_offset + i) * m + (mid_offset + j)
                v_idx = (mid_offset + j) * p + (out_offset)
                w_idx = (out_offset) * n + (row_offset + i)
                
                U[u_idx][col_idx] = 1
                V[v_idx][col_idx] = 1
                W[w_idx][col_idx] = 1
                col_idx += 1
                if col_idx > rank: raise ValueError("Rank exceeded")

    # Helper to add a naive 2x1x2 block
    def add_naive_2x1x2(row_offset, mid_offset, out_offset):
        nonlocal col_idx
        for i in range(2):
            for k in range(2):
                u_idx = (row_offset + i) * m + (mid_offset)
                v_idx = (mid_offset) * p + (out_offset + k)
                w_idx = (out_offset + k) * n + (row_offset + i)
                
                U[u_idx][col_idx] = 1
                V[v_idx][col_idx] = 1
                W[w_idx][col_idx] = 1
                col_idx += 1
                if col_idx > rank: raise ValueError("Rank exceeded")

    # Helper to add a naive 2x1x1 block
    def add_naive_2x1x1(row_offset, mid_offset, out_offset):
        nonlocal col_idx
        for i in range(2):
            u_idx = (row_offset + i) * m + (mid_offset)
            v_idx = (mid_offset) * p + (out_offset)
            w_idx = (out_offset) * n + (row_offset + i)
            
            U[u_idx][col_idx] = 1
            V[v_idx][col_idx] = 1
            W[w_idx][col_idx] = 1
            col_idx += 1
            if col_idx > rank: raise ValueError("Rank exceeded")

    # Helper to add Strassen 2x2x2 block
    def add_strassen_2x2x2(row_offset, mid_offset, out_offset):
        nonlocal col_idx
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to add 7 columns.
        # For each column, we define the coefficients in U, V, W.
        
        # Mapping of local indices (i,j,k) to global
        # i: 0,1 (row)
        # j: 0,1 (mid)
        # k: 0,1 (out)
        
        # We will define the 7 terms explicitly.
        # Term 1: (A00+A11) * (B00+B11)
        # U coeffs: A00:1, A11:1
        # V coeffs: B00:1, B11:1
        # W coeffs: C00:1, C01:0, C10:0, C11:1 (Wait, Strassen output combination is separate)
        
        # Actually, it's easier to just list the 7 multiplications and their contributions to C.
        # The tensor decomposition is a sum of rank-1 tensors.
        # Each rank-1 tensor corresponds to one multiplication.
        # The output C is a linear combination of these multiplications.
        
        # Let's define the 7 rank-1 tensors for Strassen.
        # T1: u1, v1, w1
        # ...
        # T7: u7, v7, w7
        
        # u1 = [1, 0, 0, 1] (A00, A01, A10, A11) -> A00+A11
        # v1 = [1, 0, 0, 1] (B00, B01, B10, B11) -> B00+B11
        # w1 = [1, 0, 0, 1] (C00, C01, C10, C11) -> C00+C11? No.
        
        # The standard Strassen decomposition is:
        # C = sum w_m * (u_m . A) * (v_m . B)
        
        # Let's use the standard basis.
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # We need to add 7 columns to U, V, W.
        # For column col_idx + 0 (m1):
        # U: a00:1, a11:1
        # V: b00:1, b11:1
        # W: c00:1, c11:1
        
        # For column col_idx + 1 (m2):
        # U: a10:1, a11:1
        # V: b00:1
        # W: c10:1, c11:-1
        
        # ... and so on.
        
        # Let's implement this carefully.
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col_idx
            # u_coeffs: dict {local_i: coeff}
            # v_coeffs: dict {local_jk: coeff}
            # w_coeffs: dict {local_ik: coeff}
            
            for li, cu in u_coeffs.items():
                if cu != 0:
                    global_i = row_offset + li
                    global_j = mid_offset + li_j # Wait, li is row, we need j too?
                    # No, u is vector in A space. A is 2x2.
                    # So u_coeffs should be indexed by (i,j).
            pass
            
        # Let's rewrite add_term to take flat indices.
        
        def add_term(u_flat, v_flat, w_flat):
            nonlocal col_idx
            # u_flat: list of 4 coeffs for A00, A01, A10, A11
            # v_flat: list of 4 coeffs for B00, B01, B10, B11
            # w_flat: list of 4 coeffs for C00, C01, C10, C11
            
            for li in range(2):
                for lj in range(2):
                    coeff = u_flat[li*2 + lj]
                    if coeff != 0:
                        global_i = row_offset + li
                        global_j = mid_offset + lj
                        U[global_i * m + global_j][col_idx] += coeff
                        
            for lj in range(2):
                for lk in range(2):
                    coeff = v_flat[lj*2 + lk]
                    if coeff != 0:
                        global_j = mid_offset + lj
                        global_k = out_offset + lk
                        V[global_j * p + global_k][col_idx] += coeff
                        
            for li in range(2):
                for lk in range(2):
                    coeff = w_flat[li*2 + lk]
                    if coeff != 0:
                        global_i = row_offset + li
                        global_k = out_offset + lk
                        W[global_k * n + global_i][col_idx] += coeff
                        
            col_idx += 1
            if col_idx > rank: raise ValueError("Rank exceeded")
            
        # m1
        add_term([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])
        # m2
        add_term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1])
        # m3
        add_term([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1])
        # m4
        add_term([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0])
        # m5
        add_term([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0])
        # m6
        add_term([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1])
        # m7
        add_term([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])

    # Apply to 2x5x6 block (rows 0,1)
    # Mid splits: [0,1], [2,3], [4]
    # Out splits: [0,1], [2,3], [4,5]
    
    mid_splits = [[0, 1], [2, 3], [4]]
    out_splits = [[0, 1], [2, 3], [4, 5]]
    
    for a_idx, mid_block in enumerate(mid_splits):
        for b_idx, out_block in enumerate(out_splits):
            mid_start = mid_block[0]
            out_start = out_block[0]
            mid_size = len(mid_block)
            out_size = len(out_block)
            
            if mid_size == 2 and out_size == 2:
                add_strassen_2x2x2(0, mid_start, out_start)
            elif mid_size == 2 and out_size == 1:
                add_naive_2x2x1(0, mid_start, out_start)
            elif mid_size == 1 and out_size == 2:
                add_naive_2x1x2(0, mid_start, out_start)
            else:
                add_naive_2x1x1(0, mid_start, out_start)
                
    # Apply to 1x5x6 block (row 2)
    # Naive 1x5x6
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col_idx] = 1
            V[j * p + k][col_idx] = 1
            W[k * n + 2][col_idx] = 1
            col_idx += 1
            if col_idx > rank: raise ValueError("Rank exceeded")

    if col_idx != rank:
        # Fill remaining with zeros if any (should be exactly 79)
        pass
        
    return {"U": U, "V": V, "W": W}
