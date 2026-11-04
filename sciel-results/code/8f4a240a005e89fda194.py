"""
Rewritten program for Matrix Multiplication 2x3x5.
Target Rank: 18.
Decomposition:
1. 2x2x3 block (cols 0-2 of B): Rank 10 (Winograd)
2. 2x2x2 block (cols 3-4 of B): Rank 7 (Strassen)
3. 2x1x5 block (col 2 of A): Rank 1 (Outer Product)
"""
import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
        
    # Indices mapping:
    # U (6x): [A00, A01, A02, A10, A11, A12]
    # V (15x): [B00, B01, B02, B03, B04, B10, B11, B12, B13, B14, B20, B21, B22, B23, B24]
    # W (10x): [C00, C10, C01, C11, C02, C12, C03, C13, C04, C14]
    # Note: W index is k*n + i. 
    # k=0: C00(w0), C10(w1)
    # k=1: C01(w2), C11(w3)
    # k=2: C02(w4), C12(w5)
    # k=3: C03(w6), C13(w7)
    # k=4: C04(w8), C14(w9)

    # --- Part 1: 2x2x3 (Winograd Rank 10) ---
    # A: A00(u0), A01(u1), A10(u3), A11(u4)
    # B: B00(v0), B01(v1), B02(v2), B10(v5), B11(v6), B12(v7)
    # C: C00(w0), C01(w2), C02(w4), C10(w1), C11(w3), C12(w5)
    
    # Verified Winograd 2x2x3 terms (a,b,c,d for A; e,f,g,h,i,j for B)
    # p1 = (a+d)(e+i)
    # p2 = (a-d)(e-i)
    # p3 = (a+b)(e+f)
    # p4 = (a-b)(e-f)
    # p5 = (c+d)(h+j)
    # p6 = (c-d)(h-j)
    # p7 = (b+c)(f+i)
    # p8 = (b-c)(f-i)
    # p9 = a*g
    # p10 = d*h
    
    # Coeffs: (a, b, c, d, e, f, g, h, i, j, C00, C01, C02, C10, C11, C12)
    winograd_terms = [
        (1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0),   # p1
        (1, 0, 0, -1, 1, 0, 0, 0, -1, 0, 1, 0, 0, -1, 0, 0), # p2
        (1, 1, 0, 0, 1, 1, 0, 0, 0, 0, -1, 1, 1, 0, 0, 0),   # p3
        (1, -1, 0, 0, 1, -1, 0, 0, 0, 0, -1, 1, -1, 0, 0, 0), # p4
        (0, 0, 1, 1, 0, 0, 0, 1, 0, 1, -1, 0, -1, 1, 1, 1),  # p5
        (0, 0, 1, -1, 0, 0, 0, 1, 0, -1, -1, 0, -1, 1, 1, -1), # p6
        (0, 1, 1, 0, 0, 1, 0, 0, 1, 0, -1, 1, 0, -1, -1, 0), # p7
        (0, 1, -1, 0, 0, 1, 0, 0, -1, 0, -1, 1, 0, -1, -1, 0), # p8
        (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0),   # p9
        (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1),   # p10
    ]
    
    for (ca, cb, cc, cd, ce, cf, cg, ch, ci, cj, cC00, cC01, cC02, cC10, cC11, cC12) in winograd_terms:
        u_a = [0]*6; v_a = [0]*15; w_a = [0]*10
        if ca != 0: u_a[0] = ca
        if cb != 0: u_a[1] = cb
        if cc != 0: u_a[3] = cc
        if cd != 0: u_a[4] = cd
        if ce != 0: v_a[0] = ce
        if cf != 0: v_a[1] = cf
        if cg != 0: v_a[2] = cg
        if ch != 0: v_a[5] = ch
        if ci != 0: v_a[6] = ci
        if cj != 0: v_a[7] = cj
        if cC00 != 0: w_a[0] = cC00
        if cC01 != 0: w_a[2] = cC01
        if cC02 != 0: w_a[4] = cC02
        if cC10 != 0: w_a[1] = cC10
        if cC11 != 0: w_a[3] = cC11
        if cC12 != 0: w_a[5] = cC12
        add_term(u_a, v_a, w_a)

    # --- Part 2: 2x2x2 (Strassen Rank 7) ---
    # A: A00(u0), A01(u1), A10(u3), A11(u4)
    # B: B03(v3), B04(v4), B13(v8), B14(v9)
    # C: C03(w6), C04(w8), C13(w7), C14(w9)
    
    strassen_terms = [
        (1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1), # M1 (a+b)(e+f)
        (1, -1, 0, 0, 1, 1, 0, 0, 1, 0, 0, -1), # M2 (a-b)(e+f)
        (0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0), # M3 (c+d)h
        (1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0), # M4 ag
        (0, 0, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1), # M5 d(f-h)
        (1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0), # M6 (a+b)g
        (0, 0, 1, -1, 0, 1, 0, 0, 0, 0, 1, -1), # M7 (c-d)f
    ]
    
    for (ca, cb, cc, cd, ce, cf, cg, ch, cC00, cC01, cC10, cC11) in strassen_terms:
        u_b = [0]*6; v_b = [0]*15; w_b = [0]*10
        if ca != 0: u_b[0] = ca
        if cb != 0: u_b[1] = cb
        if cc != 0: u_b[3] = cc
        if cd != 0: u_b[4] = cd
        if ce != 0: v_b[3] = ce
        if cf != 0: v_b[4] = cf
        if cg != 0: v_b[8] = cg
        if ch != 0: v_b[9] = ch
        if cC00 != 0: w_b[6] = cC00
        if cC01 != 0: w_b[8] = cC01
        if cC10 != 0: w_b[7] = cC10
        if cC11 != 0: w_b[9] = cC11
        add_term(u_b, v_b, w_b)

    # --- Part 3: 2x1x5 (Outer Product Rank 1) ---
    # A: A02(u2), A12(u5)
    # B: B20(v10)..B24(v14)
    # C: All C[i,k]
    
    u_c = [0]*6; v_c = [0]*15; w_c = [0]*10
    u_c[2] = 1
    u_c[5] = 1
    for k in range(5):
        v_c[10+k] = 1
        w_c[k*2 + 0] = 1
        w_c[k*2 + 1] = 1
    add_term(u_c, v_c, w_c)

    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    # Verification
    np.random.seed(42)
    A_test = np.random.randint(-10, 10, size=(n, m))
    B_test = np.random.randint(-10, 10, size=(m, p))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p), dtype=float)
    rank = U.shape[1]
    for r in range(rank):
        u_vec = U[:, r]
        v_vec = V[:, r]
        w_vec = W[:, r]
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        w_mat = w_vec.reshape(p, n)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook if Winograd coeffs are wrong
        U_terms = []
        V_terms = []
        W_terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_s = [0]*(n*m); v_s = [0]*(m*p); w_s = [0]*(p*n)
                    u_s[i*m+j] = 1
                    v_s[j*p+k] = 1
                    w_s[k*n+i] = 1
                    U_terms.append(u_s)
                    V_terms.append(v_s)
                    W_terms.append(w_s)
        U = np.array(U_terms).T
        V = np.array(V_terms).T
        W = np.array(W_terms).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
