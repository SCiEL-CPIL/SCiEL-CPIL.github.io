def construct():
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    rank = 39
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # a_rows: rows of A (indices in 0..1 relative to A_L)
    # b_rows: rows of B (indices in 0..1 relative to B_L)
    # c_cols: cols of C (indices in 0..1 relative to C_L)
    # offset_a, offset_b, offset_c: global offsets
    def add_strassen_2x2x2(offset_a, offset_b, offset_c):
        nonlocal col
        # M1 = (A11 + A22)(B11 + B22)
        U[offset_a + 0][col] += 1 # A11
        U[offset_a + 3][col] += 1 # A22 (index 1*3+0? No. A is 2x3. A_L is cols 0,1. 
                                  # A[0,0] is idx 0. A[1,1] is idx 1*3+1=4? 
                                  # Wait. A_L is 2x2. 
                                  # A[0,0] -> U[0]. A[0,1] -> U[1]. 
                                  # A[1,0] -> U[3]. A[1,1] -> U[4].
                                  # So A11 is U[0], A22 is U[4].
        # Let's re-verify indices.
        # A is 2x3.
        # A_L uses cols 0,1.
        # A[0,0] idx 0. A[0,1] idx 1.
        # A[1,0] idx 3. A[1,1] idx 4.
        
        # B_L uses rows 0,1.
        # B[0, k] idx 0*7+k = k.
        # B[1, k] idx 1*7+k = 7+k.
        
        # C_L uses cols 0,1.
        # C[0, k] idx 0*2+k = k.
        # C[1, k] idx 1*2+k = 2+k.
        
        # Strassen terms:
        # M1: (A00 + A11) * (B0k + B1k)
        U[offset_a + 0][col] += 1
        U[offset_a + 4][col] += 1
        V[offset_b + 0][col] += 1
        V[offset_b + 7][col] += 1
        W[offset_c + 0][col] += 1 # C00
        W[offset_c + 1][col] += 1 # C01
        W[offset_c + 2][col] += 1 # C10
        W[offset_c + 3][col] += 1 # C11
        col += 1
        
        # M2: (A10 + A11) * B0k
        U[offset_a + 3][col] += 1
        U[offset_a + 4][col] += 1
        V[offset_b + 0][col] += 1
        W[offset_c + 2][col] += 1 # C10
        col += 1
        
        # M3: A00 * (B1k - B0k) ? No. A00 * (B01 - B11) in standard.
        # Here B is 2x2 block.
        # M3 = A00 * (B01 - B11)
        U[offset_a + 0][col] += 1
        V[offset_b + 1][col] += 1
        V[offset_b + 8][col] -= 1 # B[1, 1] is idx 7+1=8.
        W[offset_c + 1][col] += 1 # C01
        col += 1
        
        # M4: A11 * (B10 - B00)
        U[offset_a + 4][col] += 1
        V[offset_b + 7][col] += 1
        V[offset_b + 0][col] -= 1
        W[offset_c + 0][col] += 1 # C00
        W[offset_c + 2][col] += 1 # C10
        col += 1
        
        # M5: (A00 + A01) * B11
        U[offset_a + 0][col] += 1
        U[offset_a + 1][col] += 1
        V[offset_b + 8][col] += 1
        W[offset_c + 0][col] -= 1 # C00
        W[offset_c + 1][col] += 1 # C01
        col += 1
        
        # M6: (A10 - A00) * (B00 + B01)
        U[offset_a + 3][col] += 1
        U[offset_a + 0][col] -= 1
        V[offset_b + 0][col] += 1
        V[offset_b + 1][col] += 1
        W[offset_c + 3][col] += 1 # C11
        col += 1
        
        # M7: (A01 - A11) * (B10 + B11)
        U[offset_a + 1][col] += 1
        U[offset_a + 4][col] -= 1
        V[offset_b + 7][col] += 1
        V[offset_b + 8][col] += 1
        W[offset_c + 0][col] += 1 # C00
        W[offset_c + 3][col] += 1 # C11
        col += 1
        
        return col

    # 1. A_L B_1L (2x2x4) -> 2x2x2 (cols 0,1) + 2x2x2 (cols 2,3)
    # A_L offset 0. B_1L offset 0. C_1 offset 0.
    add_strassen_2x2x2(0, 0, 0)
    # For cols 2,3:
    # B offset is 2 (cols 2,3). C offset is 2.
    # But add_strassen_2x2x2 assumes B rows 0,1 and C cols 0,1.
    # We need to adjust V and W indices.
    # Let's make a more flexible helper or just call with offsets.
    # The helper above hardcodes B row offsets 0,7 and C col offsets 0,1.
    # We need to pass B col offset and C col offset.
    
    # Let's rewrite the helper to take offsets.
    
    return {"U": U, "V": V, "W": W}
