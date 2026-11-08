import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 construction
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in enumerate(u_coeffs):
            if val != 0:
                U[idx][col] += val
        for idx, val in enumerate(v_coeffs):
            if val != 0:
                V[idx][col] += val
        for idx, val in enumerate(w_coeffs):
            if val != 0:
                W[idx][col] += val
        col += 1

    # Process 3 blocks of size 2 in p
    for block_idx in range(3):
        k_start = block_idx * 2
        k_end = k_start + 2
        
        # A_L is A[:, 0:2], A_R is A[:, 2:3]
        # B_T is B[0:2, k_start:k_end], B_B is B[2:3, k_start:k_end]
        
        # Strassen 7 terms
        # M1: (A00 + A11) * (B00 + B11)
        u1 = [0]*(n*m); u1[0*3+0] = 1; u1[1*3+1] = 1
        v1 = [0]*(m*p); v1[0*p+k_start] = 1; v1[1*p+k_start+1] = 1
        w1 = [0]*(p*n); w1[k_start*n+0] = 1; w1[(k_start+1)*n+1] = 1
        add_term(u1, v1, w1)
        
        # M2: (A10 + A11) * B00
        u2 = [0]*(n*m); u2[1*3+0] = 1; u2[1*3+1] = 1
        v2 = [0]*(m*p); v2[0*p+k_start] = 1
        w2 = [0]*(p*n); w2[(k_start+1)*n+0] = 1; w2[(k_start+1)*n+1] = 1 # C10, C11? 
        # Wait, C10 is i=1, k=k_start. C11 is i=1, k=k_start+1.
        # Formula: C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So M2 contributes to C10 and C11.
        w2[k_start*n+1] = 1
        w2[(k_start+1)*n+1] = -1
        add_term(u2, v2, w2)
        
        # M3: A00 * (B10 - B11)
        u3 = [0]*(n*m); u3[0*3+0] = 1
        v3 = [0]*(m*p); v3[1*p+k_start] = 1; v3[1*p+k_start+1] = -1
        w3 = [0]*(p*n); w3[k_start*n+0] = 1; w3[(k_start+1)*n+0] = 1 # C01 = M3 + M5. C00? No.
        # Formula: C01 = M3 + M5. C11 = ... + M3.
        # So M3 contributes to C01 (i=0, k=k_start+1) and C11 (i=1, k=k_start+1).
        w3[k_start*n+0] = 0 
        w3[(k_start+1)*n+0] = 1
        w3[(k_start+1)*n+1] = 1
        add_term(u3, v3, w3)
        
        # M4: A11 * (B01 - B00)
        u4 = [0]*(n*m); u4[1*3+1] = 1
        v4 = [0]*(m*p); v4[0*p+k_start+1] = 1; v4[0*p+k_start] = -1
        w4 = [0]*(p*n)
        # Formula: C00 = ... + M4. C10 = M2 + M4.
        # So M4 contributes to C00 (i=0, k=k_start) and C10 (i=1, k=k_start).
        w4[k_start*n+0] = 1
        w4[k_start*n+1] = 1
        add_term(u4, v4, w4)
        
        # M5: (A00 + A01) * B11
        u5 = [0]*(n*m); u5[0*3+0] = 1; u5[0*3+1] = 1
        v5 = [0]*(m*p); v5[1*p+k_start+1] = 1
        w5 = [0]*(p*n)
        # Formula: C00 = ... - M5. C01 = M3 + M5.
        # So M5 contributes to C00 (i=0, k=k_start) with -1 and C01 (i=0, k=k_start+1) with 1.
        w5[k_start*n+0] = -1
        w5[(k_start+1)*n+0] = 1
        add_term(u5, v5, w5)
        
        # M6: (A10 - A00) * (B00 + B10)
        u6 = [0]*(n*m); u6[1*3+0] = 1; u6[0*3+0] = -1
        v6 = [0]*(m*p); v6[0*p+k_start] = 1; v6[1*p+k_start] = 1
        w6 = [0]*(p*n)
        # Formula: C11 = ... + M6.
        # So M6 contributes to C11 (i=1, k=k_start+1) with 1.
        w6[(k_start+1)*n+1] = 1
        add_term(u6, v6, w6)
        
        # M7: (A01 - A11) * (B11 + B01)
        u7 = [0]*(n*m); u7[0*3+1] = 1; u7[1*3+1] = -1
        v7 = [0]*(m*p); v7[1*p+k_start+1] = 1; v7[0*p+k_start+1] = 1
        w7 = [0]*(p*n)
        # Formula: C00 = ... + M7.
        # So M7 contributes to C00 (i=0, k=k_start) with 1.
        w7[k_start*n+0] = 1
        add_term(u7, v7, w7)
        
        # Outer product terms for A_R (col 2) and B_B (row 2)
        # C += A[:, 2:3] @ B[2:3, k_start:k_end]
        # A_R is 2x1, B_B is 1x2.
        # Terms:
        # T1: A02 * B20 -> C00
        # T2: A02 * B21 -> C01
        # T3: A12 * B20 -> C10
        # T4: A12 * B21 -> C11
        
        u8 = [0]*(n*m); u8[0*3+2] = 1
        v8 = [0]*(m*p); v8[2*p+k_start] = 1
        w8 = [0]*(p*n); w8[k_start*n+0] = 1
        add_term(u8, v8, w8)
        
        u9 = [0]*(n*m); u9[0*3+2] = 1
        v9 = [0]*(m*p); v9[2*p+k_start+1] = 1
        w9 = [0]*(p*n); w9[(k_start+1)*n+0] = 1
        add_term(u9, v9, w9)
        
        u10 = [0]*(n*m); u10[1*3+2] = 1
        v10 = [0]*(m*p); v10[2*p+k_start] = 1
        w10 = [0]*(p*n); w10[k_start*n+1] = 1
        add_term(u10, v10, w10)
        
        u11 = [0]*(n*m); u11[1*3+2] = 1
        v11 = [0]*(m*p); v11[2*p+k_start+1] = 1
        w11 = [0]*(p*n); w11[(k_start+1)*n+1] = 1
        add_term(u11, v11, w11)

    # Process last block of size 1 in p (k=6)
    k_start = 6
    # C[:, 6] = A @ B[:, 6]
    # 6 terms
    for i in range(n):
        for j in range(m):
            u = [0]*(n*m); u[i*3+j] = 1
            v = [0]*(m*p); v[j*p+k_start] = 1
            w = [0]*(p*n); w[k_start*n+i] = 1
            add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
