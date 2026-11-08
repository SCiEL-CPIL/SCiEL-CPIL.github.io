def construct():
    import time
    start_time = time.time()
    
    n, m, p = 2, 3, 4
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
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
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1: (A00+A11)(B00+B11)
        u = {0*m + j_start: 1, 1*m + (j_start+1): 1}
        v = {j_start*p + k_start: 1, (j_start+1)*p + (k_start+1): 1}
        w = {k_start*n + 0: 1, (k_start+1)*n + 1: 1}
        add_term(u, v, w)
        
        # M2: (A10+A11)B00
        u = {1*m + j_start: 1, 1*m + (j_start+1): 1}
        v = {j_start*p + k_start: 1}
        w = {k_start*n + 1: 1}
        add_term(u, v, w)
        
        # M3: A00(B01-B11)
        u = {0*m + j_start: 1}
        v = {j_start*p + (k_start+1): 1, (j_start+1)*p + (k_start+1): -1}
        w = {(k_start+1)*n + 0: 1}
        add_term(u, v, w)
        
        # M4: A11(B10-B00)
        u = {1*m + (j_start+1): 1}
        v = {(j_start+1)*p + k_start: 1, j_start*p + k_start: -1}
        w = {k_start*n + 1: 1, (k_start+1)*n + 1: 1}
        # Note: Standard Strassen C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
        # My W for M4 adds to C10 (k_start, i=1) and C11 (k_start+1, i=1)?
        # W index is k*n + i.
        # k_start, i=1 -> k_start*n + 1.
        # k_start+1, i=1 -> (k_start+1)*n + 1.
        # This matches C10 and C11 contributions for i=1.
        add_term(u, v, w)
        
        # M5: (A00+A10)B11
        u = {0*m + j_start: 1, 1*m + j_start: 1}
        v = {(j_start+1)*p + (k_start+1): 1}
        w = {k_start*n + 0: -1, (k_start+1)*n + 0: 1}
        # C00 = ... - M5. C01 = ... + M5.
        # W for M5:
        # k_start, i=0 -> -1.
        # k_start+1, i=0 -> 1.
        add_term(u, v, w)
        
        # M6: (A11-A01)(B00+B10)
        u = {1*m + (j_start+1): 1, 0*m + (j_start+1): -1}
        v = {j_start*p + k_start: 1, (j_start+1)*p + k_start: 1}
        w = {(k_start+1)*n + 1: 1}
        # C11 = ... + M6.
        # W for M6:
        # k_start+1, i=1 -> 1.
        add_term(u, v, w)
        
        # M7: (A01-A11)(B11+B01)
        u = {0*m + (j_start+1): 1, 1*m + (j_start+1): -1}
        v = {(j_start+1)*p + (k_start+1): 1, j_start*p + (k_start+1): 1}
        w = {k_start*n + 0: 1}
        # C00 = ... + M7.
        # W for M7:
        # k_start, i=0 -> 1.
        add_term(u, v, w)
        
    # Block 1: j in 0..1, k in 0..1
    add_strassen(0, 0)
    
    # Block 2: j in 0..1, k in 2..3
    add_strassen(0, 2)
    
    # Block 3: j=2, k in 0..3
    # Term: A_i2 B_2k
    u = {0*m + 2: 1, 1*m + 2: 1}
    v = {2*p + 0: 1, 2*p + 1: 1, 2*p + 2: 1, 2*p + 3: 1}
    w = {0*n + 0: 1, 1*n + 0: 1, 2*n + 0: 1, 3*n + 0: 1,
         0*n + 1: 1, 1*n + 1: 1, 2*n + 1: 1, 3*n + 1: 1}
    add_term(u, v, w)
    
    # Check time
    if time.time() - start_time > 100:
        return {'U': U, 'V': V, 'W': W}
        
    return {'U': U, 'V': V, 'W': W}
