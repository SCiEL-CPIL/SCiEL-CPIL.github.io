import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0

    # Try Laderman's 21-term algorithm for 3x3 blocks
    # 6x6 split into 2x2 blocks of 3x3
    # 8 block products, each rank 21, total rank 168
    rank_laderman = 8 * 21  # 168
    
    U = np.zeros((n * m, rank_laderman), dtype=int)
    V = np.zeros((m * p, rank_laderman), dtype=int)
    W = np.zeros((p * n, rank_laderman), dtype=int)
    
    # Laderman's 21 terms for 3x3 matrix multiplication
    # Using 0-indexed: a[i][j], b[i][j], c[i][j] for i,j in 0,1,2
    # 
    # m1  = a[0][0] * (b[0][1] - b[1][1])
    # m2  = (a[0][0] + a[0][1]) * b[1][1]
    # m3  = a[1][1] * (b[0][0] - b[1][0])
    # m4  = (a[1][0] + a[1][1]) * b[1][0]
    # m5  = a[2][2] * (b[0][0] + b[1][1])
    # m6  = (a[0][2] - a[2][2]) * (b[0][0] + b[1][1])
    # m7  = (a[2][2] + a[1][1]) * (b[1][0] - b[0][0])
    # m8  = (a[0][0] + a[1][1]) * b[0][1]
    # m9  = a[0][2] * (b[1][1] + b[2][1])
    # m10 = (a[0][2] + a[0][0]) * (b[2][1] - b[1][1])
    # m11 = (a[0][2] + a[1][2]) * b[2][0]
    # m12 = a[2][0] * (b[0][2] + b[1][2])
    # m13 = (a[2][0] - a[2][2]) * (b[1][2] - b[0][2])
    # m14 = (a[1][0] + a[2][0]) * (b[0][2] - b[2][2])
    # m15 = (a[1][0] - a[0][0]) * (b[2][2] + b[0][2])
    # m16 = a[1][2] * (b[2][0] - b[0][0])
    # m17 = (a[1][2] + a[2][2]) * (b[0][0] - b[2][0])
    # m18 = (a[0][1] + a[1][1]) * (b[2][2] - b[1][2])
    # m19 = (a[0][1] - a[2][1]) * (b[2][2] + b[1][2])
    # m20 = a[2][1] * (b[0][1] + b[1][1])
    # m21 = (a[2][1] + a[1][1]) * (b[1][1] - b[0][1])
    #
    # c[0][0] = m5 - m6 + m7 - m15
    # c[0][1] = m1 - m8 + m10 - m17
    # c[0][2] = m12 + m13 - m14 + m19
    # c[1][0] = m3 - m4 + m16 - m18
    # c[1][1] = m2 - m3 + m4 - m5 + m8 + m17 + m18
    # c[1][2] = m9 - m11 + m14 - m19
    # c[2][0] = m6 - m7 + m15 - m16
    # c[2][1] = m8 - m10 + m17 - m20
    # c[2][2] = m9 - m12 + m13 + m14 + m16 - m18 + m20 - m21
    
    # Define Laderman terms: (U_coeff, V_coeff, W_coeffs)
    # For each term, we specify which A entries, B entries, and C entries are involved
    
    # Term definitions: (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
    # 0-indexed within 3x3 block
    laderman_terms = [
        # m1: a00*(b01-b11)
        ([(0,0)], [1], [(0,1),(1,1)], [1,-1], [(0,1)], [1]),
        # m2: (a00+a01)*b11
        ([(0,0),(0,1)], [1,1], [(1,1)], [1], [(1,1)], [1]),
        # m3: a11*(b00-b10)
        ([(1,1)], [1], [(0,0),(1,0)], [1,-1], [(1,0)], [1]),
        # m4: (a10+a11)*b10
        ([(1,0),(1,1)], [1,1], [(1,0)], [1], [(1,0)], [-1]),
        # m5: a22*(b00+b11)
        ([(2,2)], [1], [(0,0),(1,1)], [1,1], [(0,0)], [1]),
        # m6: (a02-a22)*(b00+b11)
        ([(0,2),(2,2)], [1,-1], [(0,0),(1,1)], [1,1], [(0,0)], [-1]),
        # m7: (a22+a11)*(b10-b00)
        ([(2,2),(1,1)], [1,1], [(1,0),(0,0)], [1,-1], [(0,0)], [1]),
        # m8: (a00+a11)*b01
        ([(0,0),(1,1)], [1,1], [(0,1)], [1], [(0,1)], [-1]),
        # m9: a02*(b11+b21)
        ([(0,2)], [1], [(1,1),(2,1)], [1,1], [(1,2)], [1]),
        # m10: (a02+a00)*(b21-b11)
        ([(0,2),(0,0)], [1,1], [(2,1),(1,1)], [1,-1], [(0,1)], [1]),
        # m11: (a02+a12)*b20
        ([(0,2),(1,2)], [1,1], [(2,0)], [1], [(1,2)], [-1]),
        # m12: a20*(b02+b12)
        ([(2,0)], [1], [(0,2),(1,2)], [1,1], [(0,2)], [1]),
        # m13: (a20-a22)*(b12-b02)
        ([(2,0),(2,2)], [1,-1], [(1,2),(0,2)], [1,-1], [(0,2)], [1]),
        # m14: (a10+a20)*(b02-b22)
        ([(1,0),(2,0)], [1,1], [(0,2),(2,2)], [1,-1], [(0,2)], [-1]),
        # m15: (a10-a00)*(b22+b02)
        ([(1,0),(0,0)], [1,-1], [(2,2),(0,2)], [1,1], [(0,0)], [-1]),
        # m16: a12*(b20-b00)
        ([(1,2)], [1], [(2,0),(0,0)], [1,-1], [(1,0)], [1]),
        # m17: (a12+a22)*(b00-b20)
        ([(1,2),(2,2)], [1,1], [(0,0),(2,0)], [1,-1], [(0,1)], [-1]),
        # m18: (a01+a11)*(b22-b12)
        ([(0,1),(1,1)], [1,1], [(2,2),(1,2)], [1,-1], [(1,0)], [-1]),
        # m19: (a01-a21)*(b22+b12)
        ([(0,1),(2,1)], [1,-1], [(2,2),(1,2)], [1,1], [(0,2)], [1]),
        # m20: a21*(b01+b11)
        ([(2,1)], [1], [(0,1),(1,1)], [1,1], [(2,1)], [-1]),
        # m21: (a21+a11)*(b11-b01)
        ([(2,1),(1,1)], [1,1], [(1,1),(0,1)], [1,-1], [(2,2)], [-1]),
    ]
    
    idx = 0
    for bi in range(2):  # block row of A, block col of C
        for bk in range(2):  # block col of A, block row of B
            for bj in range(2):  # block col of B, block col of C
                for t in range(21):
                    a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs = laderman_terms[t]
                    
                    # A block (bi, bk): rows 3*bi to 3*bi+2, cols 3*bk to 3*bk+2
                    # Flattened in A (6x6): (3*bi + r)*6 + (3*bk + c)
                    for ai, ac in zip(a_indices, a_coeffs):
                        row = 3*bi + ai[0]
                        col = 3*bk + ai[1]
                        U[row * 6 + col, idx] = ac
                    
                    # B block (bk, bj): rows 3*bk to 3*bk+2, cols 3*bj to 3*bj+2
                    # Flattened in B (6x6): (3*bk + r)*6 + (3*bj + c)
                    for bi2, bc in zip(b_indices, b_coeffs):
                        row = 3*bk + bi2[0]
                        col = 3*bj + bi2[1]
                        V[row * 6 + col, idx] = bc
                    
                    # C block (bi, bj): rows 3*bi to 3*bi+2, cols 3*bj to 3*bj+2
                    # Trace convention: W[c*6+r, idx]
                    for ci, cc in zip(c_indices, c_coeffs):
                        row = 3*bi + ci[0]
                        col = 3*bj + ci[1]
                        W[col * 6 + row, idx] = cc
                    
                    idx += 1
    
    assert idx == rank_laderman, f"Expected {rank_laderman}, got {idx}"
    
    # Verify Laderman
    if time.time() < deadline - 10:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = U.astype(float)
        V_arr = V.astype(float)
        W_arr = W.astype(float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                total = 0.0
                for t in range(rank_laderman):
                    s = 0.0
                    for k in range(m):
                        s += U_arr[i*m+k, t] * V_arr[k*p+j, t]
                    total += W_arr[j*n+i, t] * s
                C_computed[i, j] = total
        
        if np.allclose(C_computed, C_expected, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Strassen 2x2 blocks (rank 189)
    rank_strassen = 189
    U_s = np.zeros((n * m, rank_strassen), dtype=int)
    V_s = np.zeros((m * p, rank_strassen), dtype=int)
    W_s = np.zeros((p * n, rank_strassen), dtype=int)
    
    idx_s = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    c00 = (2*j) * 6 + 2*i
                    c01 = (2*j+1) * 6 + 2*i
                    c10 = (2*j) * 6 + (2*i+1)
                    c11 = (2*j+1) * 6 + (2*i+1)
                    
                    if s == 0:
                        U_s[a00, idx_s] = 1; U_s[a11, idx_s] = 1
                        V_s[b00, idx_s] = 1; V_s[b11, idx_s] = 1
                        W_s[c00, idx_s] = 1; W_s[c11, idx_s] = 1
                    elif s == 1:
                        U_s[a10, idx_s] = 1; U_s[a11, idx_s] = 1
                        V_s[b00, idx_s] = 1
                        W_s[c10, idx_s] = 1; W_s[c11, idx_s] = -1
                    elif s == 2:
                        U_s[a00, idx_s] = 1
                        V_s[b01, idx_s] = 1; V_s[b11, idx_s] = -1
                        W_s[c01, idx_s] = 1; W_s[c11, idx_s] = 1
                    elif s == 3:
                        U_s[a11, idx_s] = 1
                        V_s[b10, idx_s] = 1; V_s[b00, idx_s] = -1
                        W_s[c00, idx_s] = 1; W_s[c10, idx_s] = 1
                    elif s == 4:
                        U_s[a00, idx_s] = 1; U_s[a01, idx_s] = 1
                        V_s[b11, idx_s] = 1
                        W_s[c00, idx_s] = -1; W_s[c01, idx_s] = 1
                    elif s == 5:
                        U_s[a10, idx_s] = 1; U_s[a00, idx_s] = -1
                        V_s[b00, idx_s] = 1; V_s[b01, idx_s] = 1
                        W_s[c11, idx_s] = 1
                    elif s == 6:
                        U_s[a01, idx_s] = 1; U_s[a11, idx_s] = -1
                        V_s[b10, idx_s] = 1; V_s[b11, idx_s] = 1
                        W_s[c00, idx_s] = 1
                    
                    idx_s += 1
    
    assert idx_s == rank_strassen
    
    # Verify Strassen
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = U_s.astype(float)
        V_arr = V_s.astype(float)
        W_arr = W_s.astype(float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                total = 0.0
                for t in range(rank_strassen):
                    s = 0.0
                    for k in range(m):
                        s += U_arr[i*m+k, t] * V_arr[k*p+j, t]
                    total += W_arr[j*n+i, t] * s
                C_computed[i, j] = total
        
        if np.allclose(C_computed, C_expected, atol=1e-9):
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Final fallback: schoolbook
    rank_school = n * m * p
    U_f = np.zeros((n * m, rank_school), dtype=int)
    V_f = np.zeros((m * p, rank_school), dtype=int)
    W_f = np.zeros((p * n, rank_school), dtype=int)
    
    idx_f = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_f[i * m + k, idx_f] = 1
                V_f[k * p + j, idx_f] = 1
                W_f[j * n + i, idx_f] = 1
                idx_f += 1
    
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
# EVOLVE-BLOCK-END
