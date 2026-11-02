import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for i in range(n * m):
            if u_vals[i] != 0:
                U[i][col] += u_vals[i]
        for i in range(m * p):
            if v_vals[i] != 0:
                V[i][col] += v_vals[i]
        for i in range(p * n):
            if w_vals[i] != 0:
                W[i][col] += w_vals[i]
        col += 1

    # Helper to create zero arrays
    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)

    # Indices for A (U): i*m + j
    # Indices for B (V): j*p + k
    # Indices for C (W): k*n + i

    # Block 1: A_2 B_2 (Rank 10)
    # A_2 is col 2 of A (j=2)
    # B_2 is row 2 of B (j=2)
    # C_2 is all of C
    for i in range(n):
        for k in range(p):
            u = zero_u(); v = zero_v(); w = zero_w()
            u[i * m + 2] = 1
            v[2 * p + k] = 1
            w[k * n + i] = 1
            add_term(u, v, w)

    # Block 2: A_1 B_1a (Rank 7)
    # A_1 is cols 0,1 of A
    # B_1a is cols 0,1 of B_1 (rows 0,1 of B)
    # C_1a is cols 0,1 of C
    # Strassen mapping
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    
    def add_strassen(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, c11_idx, c12_idx, c21_idx, c22_idx):
        nonlocal col
        # M1 = (a+d)(e+h)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_idx] = 1; u[d_idx] = 1
        v[e_idx] = 1; v[h_idx] = 1
        w[c11_idx] = 1; w[c22_idx] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = zero_u(); v = zero_v(); w = zero_w()
        u[c_idx] = 1; u[d_idx] = 1
        v[e_idx] = 1
        w[c21_idx] = 1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_idx] = 1
        v[f_idx] = 1; v[h_idx] = -1
        w[c12_idx] = 1; w[c22_idx] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[d_idx] = 1
        v[g_idx] = 1; v[e_idx] = -1
        w[c11_idx] = 1; w[c21_idx] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_idx] = 1; u[b_idx] = 1
        v[h_idx] = 1
        w[c11_idx] = -1; w[c12_idx] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[c_idx] = 1; u[a_idx] = -1
        v[e_idx] = 1; v[f_idx] = 1
        w[c22_idx] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = zero_u(); v = zero_v(); w = zero_w()
        u[b_idx] = 1; u[d_idx] = -1
        v[g_idx] = 1; v[h_idx] = 1
        w[c11_idx] = 1
        add_term(u, v, w)

    # Indices for Block 2
    # A: 0,1 -> 0,1; 1,0 -> 3; 1,1 -> 4
    a_idx = 0 * m + 0
    b_idx = 0 * m + 1
    c_idx = 1 * m + 0
    d_idx = 1 * m + 1
    # B: 0,0 -> 0; 0,1 -> 1; 1,0 -> 5; 1,1 -> 6
    e_idx = 0 * p + 0
    f_idx = 0 * p + 1
    g_idx = 1 * p + 0
    h_idx = 1 * p + 1
    # C: 0,0 -> 0; 0,1 -> 2; 1,0 -> 1; 1,1 -> 3
    c11_idx = 0 * n + 0
    c12_idx = 1 * n + 0
    c21_idx = 0 * n + 1
    c22_idx = 1 * n + 1
    
    add_strassen(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, c11_idx, c12_idx, c21_idx, c22_idx)

    # Block 3: A_1 B_1b (Rank 7)
    # B_1b is cols 2,3 of B (rows 0,1)
    # C_1b is cols 2,3 of C
    e_idx = 0 * p + 2
    f_idx = 0 * p + 3
    g_idx = 1 * p + 2
    h_idx = 1 * p + 3
    c11_idx = 2 * n + 0
    c12_idx = 3 * n + 0
    c21_idx = 2 * n + 1
    c22_idx = 3 * n + 1
    
    add_strassen(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, c11_idx, c12_idx, c21_idx, c22_idx)

    # Block 4: A_1 B_1c (Rank 4)
    # B_1c is col 4 of B (rows 0,1)
    # C_1c is col 4 of C
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    
    # Term 1: A[0,0] B[0,4] -> C[0,4]
    u = zero_u(); v = zero_v(); w = zero_w()
    u[0 * m + 0] = 1
    v[0 * p + 4] = 1
    w[4 * n + 0] = 1
    add_term(u, v, w)
    
    # Term 2: A[0,1] B[1,4] -> C[0,4]
    u = zero_u(); v = zero_v(); w = zero_w()
    u[0 * m + 1] = 1
    v[1 * p + 4] = 1
    w[4 * n + 0] = 1
    add_term(u, v, w)
    
    # Term 3: A[1,0] B[0,4] -> C[1,4]
    u = zero_u(); v = zero_v(); w = zero_w()
    u[1 * m + 0] = 1
    v[0 * p + 4] = 1
    w[4 * n + 1] = 1
    add_term(u, v, w)
    
    # Term 4: A[1,1] B[1,4] -> C[1,4]
    u = zero_u(); v = zero_v(); w = zero_w()
    u[1 * m + 1] = 1
    v[1 * p + 4] = 1
    w[4 * n + 1] = 1
    add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
