import numpy as np

def construct():
    n, m, p = 2, 3, 5
    total_rank = 19
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    r = 0
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal r
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
        r += 1

    # Block 1: A_2 B_2 (2x1 by 1x5)
    # A_2 is A[:, 2]
    # B_2 is B[2, :]
    u1 = np.zeros(n * m)
    u1[0 * m + 2] = 1.0 # i=0, k=2
    u1[1 * m + 2] = 1.0 # i=1, k=2
    
    v1 = np.zeros(m * p)
    for j in range(p):
        v1[2 * p + j] = 1.0 # k=2, j
        
    w1 = np.ones(p * n)
    add_term(u1, v1, w1)
    
    # Block 2: A_1 B_1a (2x2 by 2x2)
    # A_1 is A[:, 0:2]
    # B_1a is B[0:2, 0:2]
    # Strassen 2x2
    # M1 = (A11+A22)(B11+B22)
    u2_1 = np.zeros(n * m)
    u2_1[0 * m + 0] = 1.0
    u2_1[1 * m + 1] = 1.0
    v2_1 = np.zeros(m * p)
    v2_1[0 * p + 0] = 1.0
    v2_1[1 * p + 1] = 1.0
    w2_1 = np.zeros(p * n)
    w2_1[0 * n + 0] = 1.0 # C00
    w2_1[1 * n + 1] = 1.0 # C11
    add_term(u2_1, v2_1, w2_1)
    
    # M2 = (A21+A22)B11
    u2_2 = np.zeros(n * m)
    u2_2[1 * m + 0] = 1.0
    u2_2[1 * m + 1] = 1.0
    v2_2 = np.zeros(m * p)
    v2_2[0 * p + 0] = 1.0
    w2_2 = np.zeros(p * n)
    w2_2[0 * n + 1] = -1.0 # C01? No, C21 and C22 in Strassen?
    # Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # Indices: C11 is (0,0), C12 is (0,1), C21 is (1,0), C22 is (1,1)
    
    # Let's re-map carefully.
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 -= 1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += 1, C10 += 1
    # M5: C00 -= 1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1
    
    # M2
    w2_2 = np.zeros(p * n)
    w2_2[0 * n + 1] = 1.0 # C10 (row 1, col 0) -> index 0*n + 1? No.
    # W index is j*n + i. j is col, i is row.
    # C10 is row 1, col 0. j=0, i=1. Index 0*n + 1 = 1.
    # C11 is row 1, col 1. j=1, i=1. Index 1*n + 1 = 3.
    w2_2[1] = 1.0
    w2_2[3] = -1.0
    add_term(u2_2, v2_2, w2_2)
    
    # M3 = A11(B12-B22)
    u2_3 = np.zeros(n * m)
    u2_3[0 * m + 0] = 1.0
    v2_3 = np.zeros(m * p)
    v2_3[0 * p + 1] = 1.0
    v2_3[1 * p + 1] = -1.0
    w2_3 = np.zeros(p * n)
    w2_3[1 * n + 0] = 1.0 # C01 (row 0, col 1). j=1, i=0. Index 1*n + 0 = 2.
    w2_3[2] = 1.0
    w2_3[3] = 1.0 # C11
    add_term(u2_3, v2_3, w2_3)
    
    # M4 = A22(B21-B11)
    u2_4 = np.zeros(n * m)
    u2_4[1 * m + 1] = 1.0
    v2_4 = np.zeros(m * p)
    v2_4[1 * p + 0] = 1.0
    v2_4[0 * p + 0] = -1.0
    w2_4 = np.zeros(p * n)
    w2_4[0 * n + 0] = 1.0 # C00
    w2_4[1 * n + 0] = 1.0 # C10
    add_term(u2_4, v2_4, w2_4)
    
    # M5 = (A11+A12)B22
    u2_5 = np.zeros(n * m)
    u2_5[0 * m + 0] = 1.0
    u2_5[0 * m + 1] = 1.0
    v2_5 = np.zeros(m * p)
    v2_5[1 * p + 1] = 1.0
    w2_5 = np.zeros(p * n)
    w2_5[0 * n + 0] = -1.0 # C00
    w2_5[1 * n + 0] = 1.0 # C01
    add_term(u2_5, v2_5, w2_5)
    
    # M6 = (A21-A11)(B11+B12)
    u2_6 = np.zeros(n * m)
    u2_6[1 * m + 0] = 1.0
    u2_6[0 * m + 0] = -1.0
    v2_6 = np.zeros(m * p)
    v2_6[0 * p + 0] = 1.0
    v2_6[0 * p + 1] = 1.0
    w2_6 = np.zeros(p * n)
    w2_6[3] = 1.0 # C11
    add_term(u2_6, v2_6, w2_6)
    
    # M7 = (A12-A22)(B21+B22)
    u2_7 = np.zeros(n * m)
    u2_7[0 * m + 1] = 1.0
    u2_7[1 * m + 1] = -1.0
    v2_7 = np.zeros(m * p)
    v2_7[1 * p + 0] = 1.0
    v2_7[1 * p + 1] = 1.0
    w2_7 = np.zeros(p * n)
    w2_7[0 * n + 0] = 1.0 # C00
    add_term(u2_7, v2_7, w2_7)
    
    # Block 3: A_1 B_1b (2x2 by 2x2)
    # B_1b is B[0:2, 2:4]
    # Same Strassen, but V indices shifted by 2 in col.
    # V index: k * p + j. j is 2,3 instead of 0,1.
    
    # M1
    u3_1 = np.zeros(n * m)
    u3_1[0 * m + 0] = 1.0
    u3_1[1 * m + 1] = 1.0
    v3_1 = np.zeros(m * p)
    v3_1[0 * p + 2] = 1.0
    v3_1[1 * p + 3] = 1.0
    w3_1 = np.zeros(p * n)
    w3_1[2 * n + 0] = 1.0 # C02
    w3_1[3 * n + 1] = 1.0 # C13
    add_term(u3_1, v3_1, w3_1)
    
    # M2
    u3_2 = np.zeros(n * m)
    u3_2[1 * m + 0] = 1.0
    u3_2[1 * m + 1] = 1.0
    v3_2 = np.zeros(m * p)
    v3_2[0 * p + 2] = 1.0
    w3_2 = np.zeros(p * n)
    w3_2[2 * n + 1] = 1.0 # C12
    w3_2[3 * n + 1] = -1.0 # C13
    add_term(u3_2, v3_2, w3_2)
    
    # M3
    u3_3 = np.zeros(n * m)
    u3_3[0 * m + 0] = 1.0
    v3_3 = np.zeros(m * p)
    v3_3[0 * p + 3] = 1.0
    v3_3[1 * p + 3] = -1.0
    w3_3 = np.zeros(p * n)
    w3_3[3 * n + 0] = 1.0 # C03
    w3_3[3 * n + 1] = 1.0 # C13
    add_term(u3_3, v3_3, w3_3)
    
    # M4
    u3_4 = np.zeros(n * m)
    u3_4[1 * m + 1] = 1.0
    v3_4 = np.zeros(m * p)
    v3_4[1 * p + 2] = 1.0
    v3_4[0 * p + 2] = -1.0
    w3_4 = np.zeros(p * n)
    w3_4[2 * n + 0] = 1.0 # C02
    w3_4[2 * n + 1] = 1.0 # C12
    add_term(u3_4, v3_4, w3_4)
    
    # M5
    u3_5 = np.zeros(n * m)
    u3_5[0 * m + 0] = 1.0
    u3_5[0 * m + 1] = 1.0
    v3_5 = np.zeros(m * p)
    v3_5[1 * p + 3] = 1.0
    w3_5 = np.zeros(p * n)
    w3_5[2 * n + 0] = -1.0 # C02
    w3_5[3 * n + 0] = 1.0 # C03
    add_term(u3_5, v3_5, w3_5)
    
    # M6
    u3_6 = np.zeros(n * m)
    u3_6[1 * m + 0] = 1.0
    u3_6[0 * m + 0] = -1.0
    v3_6 = np.zeros(m * p)
    v3_6[0 * p + 2] = 1.0
    v3_6[0 * p + 3] = 1.0
    w3_6 = np.zeros(p * n)
    w3_6[3 * n + 1] = 1.0 # C13
    add_term(u3_6, v3_6, w3_6)
    
    # M7
    u3_7 = np.zeros(n * m)
    u3_7[0 * m + 1] = 1.0
    u3_7[1 * m + 1] = -1.0
    v3_7 = np.zeros(m * p)
    v3_7[1 * p + 2] = 1.0
    v3_7[1 * p + 3] = 1.0
    w3_7 = np.zeros(p * n)
    w3_7[2 * n + 0] = 1.0 # C02
    add_term(u3_7, v3_7, w3_7)
    
    # Block 4: A_1 B_1c (2x2 by 2x1)
    # B_1c is B[0:2, 4]
    # Trivial Rank 4
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    # Term 1: A00 B04
    u4_1 = np.zeros(n * m)
    u4_1[0 * m + 0] = 1.0
    v4_1 = np.zeros(m * p)
    v4_1[0 * p + 4] = 1.0
    w4_1 = np.zeros(p * n)
    w4_1[4 * n + 0] = 1.0 # C04
    add_term(u4_1, v4_1, w4_1)
    
    # Term 2: A01 B14
    u4_2 = np.zeros(n * m)
    u4_2[0 * m + 1] = 1.0
    v4_2 = np.zeros(m * p)
    v4_2[1 * p + 4] = 1.0
    w4_2 = np.zeros(p * n)
    w4_2[4 * n + 0] = 1.0 # C04
    add_term(u4_2, v4_2, w4_2)
    
    # Term 3: A10 B04
    u4_3 = np.zeros(n * m)
    u4_3[1 * m + 0] = 1.0
    v4_3 = np.zeros(m * p)
    v4_3[0 * p + 4] = 1.0
    w4_3 = np.zeros(p * n)
    w4_3[4 * n + 1] = 1.0 # C14
    add_term(u4_3, v4_3, w4_3)
    
    # Term 4: A11 B14
    u4_4 = np.zeros(n * m)
    u4_4[1 * m + 1] = 1.0
    v4_4 = np.zeros(m * p)
    v4_4[1 * p + 4] = 1.0
    w4_4 = np.zeros(p * n)
    w4_4[4 * n + 1] = 1.0 # C14
    add_term(u4_4, v4_4, w4_4)
    
    return {'U': U, 'V': V, 'W': W}
