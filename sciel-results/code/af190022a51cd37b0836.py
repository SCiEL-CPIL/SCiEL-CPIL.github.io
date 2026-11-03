import numpy as np

DIMENSIONS = [2, 3, 7]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    # M1 = (a+d)(e+h)
    # M2 = (a+d)(f-h)
    # M3 = (c-d)e
    # M4 = (c-d)h
    # M5 = a(f-g)
    # M6 = b(e-g)
    # M7 = (a+b)h
    # C11 = M1 + M4 - M5 - M7
    # C12 = M2 + M7
    # C21 = M3 + M4
    # C22 = M1 - M2 + M3 + M5
    
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices in U (row i, col j) -> i*m+j
    # Indices in V (row j, col k) -> j*p+k
    # Indices in W (row k, col i) -> k*n+i
    
    # M1: (A11+A22)(B11+B22)
    # U: A11=1, A22=1
    U[0, 0] = 1; U[3, 0] = 1
    # V: B11=1, B22=1
    V[0, 0] = 1; V[3, 0] = 1
    # W: C11 += 1, C22 += 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A11+A22)(B12-B22)
    U[0, 1] = 1; U[3, 1] = 1
    V[1, 1] = 1; V[3, 1] = -1
    # W: C12 += 1, C22 -= 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: (A21-A22)B11
    U[2, 2] = 1; U[3, 2] = -1
    V[0, 2] = 1
    # W: C21 += 1, C22 += 1
    W[0, 2] = 1; W[3, 2] = 1
    
    # M4: (A21-A22)B22
    U[2, 3] = 1; U[3, 3] = -1
    V[3, 3] = 1
    # W: C11 -= 1, C21 += 1
    W[0, 3] = -1; W[0, 2] += 1 # Wait, W[0,2] is C21?
    # W index: k*n+i.
    # C11: k=0, i=0 -> W[0,0]
    # C12: k=1, i=0 -> W[1,0]
    # C21: k=0, i=1 -> W[0,1]
    # C22: k=1, i=1 -> W[1,1]
    
    # Let's re-map W indices carefully.
    # W[k*n + i, col]
    # C11 (i=0, k=0) -> W[0, col]
    # C12 (i=0, k=1) -> W[1, col]
    # C21 (i=1, k=0) -> W[2, col]
    # C22 (i=1, k=1) -> W[3, col]
    
    # Reset W
    W = np.zeros((p * n, rank))
    
    # M1: C11 += 1, C22 += 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: C12 += 1, C22 -= 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: C21 += 1, C22 += 1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: C11 -= 1, C21 += 1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: A11(B12-B22)
    U[0, 4] = 1
    V[1, 4] = 1; V[3, 4] = -1
    # C11 -= 1, C22 += 1
    W[0, 4] = -1; W[3, 4] = 1
    
    # M6: A12(B11-B21)
    U[1, 5] = 1
    V[0, 5] = 1; V[2, 5] = -1
    # C11 += 1, C22 -= 1
    W[0, 5] = 1; W[3, 5] = -1
    
    # M7: (A11+A12)B22
    U[0, 6] = 1; U[1, 6] = 1
    V[3, 6] = 1
    # C11 -= 1, C12 += 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W

def _embed(U_sub, V_sub, W_sub, n, m_sub, p, m_offset, n_offset, p_offset):
    # Embed sub-problem (n, m_sub, p) into (n, m_total, p) at m_offset
    # U_sub is (n*m_sub, R), V_sub is (m_sub*p, R), W_sub is (p*n, R)
    # We need to create full sized matrices with zeros and fill in the relevant parts.
    # This function is for combining in the m dimension.
    pass

def construct():
    n, m, p = DIMENSIONS
    # Split m=3 into 2 and 1
    m1, m2 = 2, 1
    
    # Part 1: 2x2x7
    # Split p=7 into 3 and 4
    p1, p2 = 3, 4
    
    # Sub-part 1.1: 2x2x3
    # Split p=3 into 2 and 1
    p11, p12 = 2, 1
    U_111, V_111, W_111 = _strassen_222()
    U_112, V_112, W_112 = _schoolbook(2, 2, 1)
    
    # Combine 1.11 and 1.12 in p dimension
    # p_total = 3
    # U is (2*2, R). Same for both.
    # V is (2*p, R). V_111 is (2*2, 7), V_112 is (2*1, 4).
    # V_111 occupies p indices 0,1. V_112 occupies p index 2.
    U_11 = np.hstack([U_111, U_112])
    V_11 = np.zeros((2 * 3, 7 + 4))
    V_11[0:4, 0:7] = V_111
    V_11[4:6, 7:11] = V_112
    W_11 = np.hstack([W_111, W_112])
    
    # Sub-part 1.2: 2x2x4
    # Split p=4 into 2 and 2
    U_121, V_121, W_121 = _strassen_222()
    U_122, V_122, W_122 = _strassen_222()
    
    U_12 = np.hstack([U_121, U_122])
    V_12 = np.zeros((2 * 4, 7 + 7))
    V_12[0:4, 0:7] = V_121
    V_12[4:8, 7:14] = V_122
    W_12 = np.hstack([W_121, W_122])
    
    # Combine 1.1 and 1.2 in p dimension
    # p_total = 7
    U_1 = np.hstack([U_11, U_12])
    V_1 = np.zeros((2 * 7, (7+4) + (7+7)))
    V_1[0:6, 0:11] = V_11
    V_1[6:14, 11:25] = V_12
    W_1 = np.hstack([W_11, W_12])
    
    # Part 2: 2x1x7
    U_2, V_2, W_2 = _schoolbook(2, 1, 7)
    
    # Combine Part 1 and Part 2 in m dimension
    # m_total = 3
    U_total = np.zeros((2 * 3, U_1.shape[1] + U_2.shape[1]))
    # U_1 is for m=2 (indices 0,1). U_2 is for m=1 (index 2).
    U_total[0:4, 0:U_1.shape[1]] = U_1
    U_total[4:6, U_1.shape[1]:] = U_2
    
    V_total = np.hstack([V_1, V_2])
    # V_1 is (2*7, R1). V_2 is (1*7, R2).
    # V_1 corresponds to m indices 0,1. V_2 corresponds to m index 2.
    # V shape is (m*p, R) = (3*7, R_total).
    # V_1 occupies rows 0..13 (m=0,1).
    # V_2 occupies rows 14..20 (m=2).
    V_total_full = np.zeros((3 * 7, V_total.shape[1]))
    V_total_full[0:14, :] = V_total[:, :V_1.shape[1]]
    V_total_full[14:21, :] = V_total[:, V_1.shape[1]:]
    
    W_total = np.hstack([W_1, W_2])
    
    return {
        "U": U_total.tolist(),
        "V": V_total_full.tolist(),
        "W": W_total.tolist()
    }
