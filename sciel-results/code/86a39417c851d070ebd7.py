def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps definitions
    # a,b,c,d are A00, A01, A10, A11
    # e,f,g,h are B00, B01, B10, B11
    
    # M1: (a+d)(e+h)
    # U: a(+1), d(+1)
    # V: e(+1), h(+1)
    # W: C00(+1), C11(+1)
    
    # M2: (c+d)e
    # U: c(+1), d(+1)
    # V: e(+1)
    # W: C10(+1), C11(-1)
    
    # M3: a(f-h)
    # U: a(+1)
    # V: f(+1), h(-1)
    # W: C01(+1), C11(+1)
    
    # M4: d(g-e)
    # U: d(+1)
    # V: g(+1), e(-1)
    # W: C00(+1), C10(+1)
    
    # M5: (a+b)h
    # U: a(+1), b(+1)
    # V: h(+1)
    # W: C00(-1), C01(+1)
    
    # M6: (c-a)(e+f)
    # U: c(+1), a(-1)
    # V: e(+1), f(+1)
    # W: C11(+1)
    
    # M7: (b-d)(g+h)
    # U: b(+1), d(-1)
    # V: g(+1), h(+1)
    # W: C00(+1)
    
    steps = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [(local_i, local_j, coeff), ...]
        # V: [(local_j, local_k, coeff), ...]
        # W: [(local_k, local_i, coeff), ...]
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]), # M1
        ([((1,0,1), (1,1,1))], [((0,0,1))], [((0,1,1), (1,1,-1))]),          # M2
        ([((0,0,1))], [((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))]),          # M3
        ([((1,1,1))], [((1,0,1), (0,0,-1))], [((0,0,1), (1,0,1))]),          # M4
        ([((0,0,1), (0,1,1))], [((1,1,1))], [((0,0,-1), (1,0,1))]),          # M5
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]),          # M6
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,0,1))])           # M7
    ]
    
    # Blocks: (i_offset, j_offset, k_offset)
    # Block 0: A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 0:2], B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 2: A[0:2, 2:4], B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 2:4], B[2:4, 2:4] -> C[0:2, 2:4]
    
    blocks = [
        (0, 0, 0), # b0
        (0, 0, 2), # b1
        (0, 2, 0), # b2
        (0, 2, 2)  # b3
    ]
    
    for b_idx, (i_off, j_off, k_off) in enumerate(blocks):
        for s_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(steps):
            col = b_idx * 7 + s_idx
            
            # Fill U
            for (li, lj, coeff) in u_coeffs:
                gi = i_off + li
                gj = j_off + lj
                U[gi * m + gj][col] += coeff
                
            # Fill V
            for (lj, lk, coeff) in v_coeffs:
                gj = j_off + lj
                gk = k_off + lk
                V[gj * p + gk][col] += coeff
                
            # Fill W
            for (lk, li, coeff) in w_coeffs:
                gk = k_off + lk
                gi = i_off + li
                W[gk * n + gi][col] += coeff
                
    return {"U": U, "V": V, "W": W}
