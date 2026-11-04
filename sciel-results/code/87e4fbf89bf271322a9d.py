import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Use Kronecker product: M_{2,4,6} = M_{2,2,2} ⊗ M_{1,2,3}
    # Rank = rank(M_{2,2,2}) * rank(M_{1,2,3}) = 7 * 6 = 42
    #
    # M_{2,2,2}: Strassen decomposition (rank 7)
    # M_{1,2,3}: schoolbook decomposition (rank 6)
    
    # Strassen for M_{2,2,2}:
    # U_strassen: shape (4, 7), V_strassen: shape (4, 7), W_strassen: shape (4, 7)
    # Indices: A is 2x2 (flat: 0,1,2,3), B is 2x2 (flat: 0,1,2,3), C is 2x2 (flat: 0,1,2,3)
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # Flat: A[i*m+j], B[j*p+k], C[k*n+i] with n=m=p=2
    # A: [A00, A01, A10, A11] = indices 0,1,2,3
    # B: [B00, B01, B10, B11] = indices 0,1,2,3
    # C: [C00, C10, C01, C11] = indices 0,1,2,3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    U_s[0, 0] += 1; U_s[3, 0] += 1
    V_s[0, 0] += 1; V_s[3, 0] += 1
    W_s[0, 0] += 1; W_s[3, 0] += 1
    
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    U_s[2, 1] += 1; U_s[3, 1] += 1
    V_s[0, 1] += 1
    W_s[1, 1] += 1; W_s[3, 1] -= 1
    
    # M3 = A00(B01-B11) -> contributes to C01, C11
    U_s[0, 2] += 1
    V_s[1, 2] += 1; V_s[3, 2] -= 1
    W_s[2, 2] += 1; W_s[3, 2] += 1
    
    # M4 = A11(B10-B00) -> contributes to C00, C10
    U_s[3, 3] += 1
    V_s[2, 3] += 1; V_s[0, 3] -= 1
    W_s[0, 3] += 1; W_s[1, 3] += 1
    
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    U_s[0, 4] += 1; U_s[1, 4] += 1
    V_s[3, 4] += 1
    W_s[0, 4] -= 1; W_s[2, 4] += 1
    
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    U_s[2, 5] += 1; U_s[0, 5] -= 1
    V_s[0, 5] += 1; V_s[1, 5] += 1
    W_s[3, 5] += 1
    
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    U_s[1, 6] += 1; U_s[3, 6] -= 1
    V_s[2, 6] += 1; V_s[3, 6] += 1
    W_s[0, 6] += 1
    
    # M_{1,2,3}: multiplying 1x2 by 2x3
    # A: 1x2 = [a0, a1] (flat indices 0, 1)
    # B: 2x3 (flat indices: B00,B01,B02,B10,B11,B12 = 0,1,2,3,4,5)
    # C: 1x3 = [c0, c1, c2] (flat indices 0, 1, 2)
    # c0 = a0*B00 + a1*B10
    # c1 = a0*B01 + a1*B11
    # c2 = a0*B02 + a1*B12
    # Rank 6: each ci needs 2 terms
    U_m = np.zeros((2, 6), dtype=int)
    V_m = np.zeros((6, 6), dtype=int)
    W_m = np.zeros((3, 6), dtype=int)
    
    # c0 = a0*B00 + a1*B10
    U_m[0, 0] = 1; V_m[0, 0] = 1; W_m[0, 0] = 1
    U_m[1, 1] = 1; V_m[3, 1] = 1; W_m[0, 1] = 1
    # c1 = a0*B01 + a1*B11
    U_m[0, 2] = 1; V_m[1, 2] = 1; W_m[1, 2] = 1
    U_m[1, 3] = 1; V_m[4, 3] = 1; W_m[1, 3] = 1
    # c2 = a0*B02 + a1*B12
    U_m[0, 4] = 1; V_m[2, 4] = 1; W_m[2, 4] = 1
    U_m[1, 5] = 1; V_m[5, 5] = 1; W_m[2, 5] = 1
    
    # Now form Kronecker product
    # M_{2,4,6} = M_{2,2,2} ⊗ M_{1,2,3}
    # n_total = 2*1 = 2, m_total = 2*2 = 4, p_total = 2*3 = 6
    # 
    # For the Kronecker product of two tensor decompositions:
    # T1 = sum_{r1} u1[r1] ⊗ v1[r1] ⊗ w1[r1]  (for M_{2,2,2})
    # T2 = sum_{r2} u2[r2] ⊗ v2[r2] ⊗ w2[r2]  (for M_{1,2,3})
    # T = T1 ⊗ T2 = sum_{r1,r2} (u1[r1]⊗u2[r2]) ⊗ (v1[r1]⊗v2[r2]) ⊗ (w1[r1]⊗w2[r2])
    #
    # Index mapping:
    # A_total is 2x4. Index (i, j) where i in {0,1}, j in {0,1,2,3}
    #   j = j1 * 2 + j2 where j1 in {0,1} (from M_{2,2,2}), j2 in {0,1} (from M_{1,2,3})
    #   Flat index in A_total: i * 4 + j = i * 4 + j1 * 2 + j2
    #   In Kronecker: A_strassen[i, j1] * A_m[j2]
    #   Flat: A_strassen flat = i*2 + j1, A_m flat = j2
    #   A_total flat = (i*2 + j1) * 2 + j2 = i*4 + j1*2 + j2 ✓
    #
    # B_total is 4x6. Index (j, k) where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    #   j = j1 * 2 + j2, k = k1 * 3 + k2
    #   Flat index in B_total: j * 6 + k = (j1*2+j2)*6 + (k1*3+k2)
    #   In Kronecker: B_strassen[j1, k1] * B_m[j2, k2]
    #   B_strassen flat = j1*2 + k1, B_m flat = j2*3 + k2
    #   B_total flat = (j1*2+k1)*3 + (j2*3+k2) = j1*6 + k1*3 + j2*3 + k2
    #   But we need: (j1*2+j2)*6 + k1*3 + k2 = j1*12 + j2*6 + k1*3 + k2
    #   These are different! The Kronecker product index mapping is:
    #   B_total[j, k] where j = j1*m2 + j2 = j1*2 + j2, k = k1*p2 + k2 = k1*3 + k2
    #   Flat B_total = j * p_total + k = (j1*2+j2)*6 + (k1*3+k2)
    #   = j1*12 + j2*6 + k1*3 + k2
    #   
    #   In the Kronecker tensor, the B index is (j1, k1) from T1 and (j2, k2) from T2
    #   The flat index in the Kronecker B space: (j1, j2) for the "row" part, (k1, k2) for the "col" part
    #   B_kronecker flat = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
    #   = (j1*2 + j2) * 6 + (k1*3 + k2) ✓
    #
    # So for the V matrix of the Kronecker product:
    # V_total[flat_B_total, r] where r = r1 * 6 + r2
    # V_total[(j1*2+j2)*6 + (k1*3+k2), r1*6+r2] = V_s[j1*2+k1, r1] * V_m[j2*3+k2, r2]
    
    rank_total = 7 * 6  # 42
    U_total = np.zeros((nm, rank_total), dtype=int)
    V_total = np.zeros((mp, rank_total), dtype=int)
    W_total = np.zeros((pn, rank_total), dtype=int)
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            # U: A_total is 2x4, flat index = i*4 + j where j = j1*2 + j2
            # U_total[i*4 + j1*2 + j2, r] = U_s[i*2 + j1, r1] * U_m[j2, r2]
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        flat_a = i * 4 + j1 * 2 + j2
                        flat_s = i * 2 + j1
                        val = U_s[flat_s, r1] * U_m[j2, r2]
                        U_total[flat_a, r] += val
            
            # V: B_total is 4x6, flat index = j*6 + k where j = j1*2 + j2, k = k1*3 + k2
            # V_total[(j1*2+j2)*6 + (k1*3+k2), r] = V_s[j1*2+k1, r1] * V_m[j2*3+k2, r2]
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            flat_b = (j1 * 2 + j2) * 6 + (k1 * 3 + k2)
                            flat_s = j1 * 2 + k1
                            flat_m = j2 * 3 + k2
                            val = V_s[flat_s, r1] * V_m[flat_m, r2]
                            V_total[flat_b, r] += val
            
            # W: C_total is 6x2 (p*n = 6*2 = 12), flat index = k*n + i where k = k1*3 + k2, i in {0,1}
            # W_total[(k1*3+k2)*2 + i, r] = W_s[k1*2+i, r1] * W_m[k2, r2]
            for k1 in range(2):
                for k2 in range(3):
                    for i in range(2):
                        flat_c = (k1 * 3 + k2) * 2 + i
                        flat_s = k1 * 2 + i
                        val = W_s[flat_s, r1] * W_m[k2, r2]
                        W_total[flat_c, r] += val
    
    # Verification
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank_total):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0
                    for r in range(rank_total):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True

    if not verify(U_total, V_total, W_total):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    
        return {"U": U, "V": V, "W": W}

    U_list = [[int(x) for x in row] for row in U_total.tolist()]
    V_list = [[int(x) for x in row] for row in V_total.tolist()]
    W_list = [[int(x) for x in row] for row in W_total.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
