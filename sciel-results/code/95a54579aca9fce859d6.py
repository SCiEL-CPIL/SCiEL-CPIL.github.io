def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
        
    # Strassen helper for 2x2x2 block
    # A indices: i in 0..1, j in j_start..j_start+1
    # B indices: j in j_start..j_start+1, k in k_start..k_start+1
    # C indices: i in 0..1, k in k_start..k_start+1
    def add_strassen(j_start, k_start):
        nonlocal col
        # A rows: i*m + j
        # B rows: j*p + k
        # W rows: k*n + i
        
        # S1: (A00+A11)(B00+B11)
        u = {0*m + j_start: 1, 1*m + (j_start+1): 1}
        v = {j_start*p + k_start: 1, (j_start+1)*p + (k_start+1): 1}
        w = {k_start*n + 0: 1, (k_start+1)*n + 1: 1}
        add_term(u, v, w)
        
        # S2: (A10+A11)B00
        u = {1*m + j_start: 1, 1*m + (j_start+1): 1}
        v = {j_start*p + k_start: 1}
        w = {k_start*n + 1: 1}
        add_term(u, v, w)
        
        # S3: A00(B01-B11)
        u = {0*m + j_start: 1}
        v = {j_start*p + (k_start+1): 1, (j_start+1)*p + (k_start+1): -1}
        w = {(k_start+1)*n + 0: 1}
        add_term(u, v, w)
        
        # S4: A11(B10-B00)
        u = {1*m + (j_start+1): 1}
        v = {(j_start+1)*p + k_start: 1, j_start*p + k_start: -1}
        w = {k_start*n + 1: 1, (k_start+1)*n + 1: 1} # C10 += S4, C11 += S4
        # Wait, C10 = S2+S4, C11 = S1-S2+S3+S6.
        # S4 contributes to C10 and C11?
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So S4 (M4) is in C00 and C10.
        # My w for S4 above was wrong.
        # Let's re-verify Strassen formulas.
        
        pass # I will rewrite this properly in the final code.
        
    # I will just hardcode the 14 Strassen terms and 8 schoolbook terms in the final code to avoid bugs in the helper.
