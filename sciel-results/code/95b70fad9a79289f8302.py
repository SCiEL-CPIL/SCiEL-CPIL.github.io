import time

def construct():
    start = time.time()
    DEADLINE = 150.0
    MARGIN = 3.0

    def time_left():
        return DEADLINE - MARGIN - (time.time() - start)

    n, m, p = 4, 4, 7

    # ---- Trivial exact decomposition (guaranteed valid, rank = n*m*p) ----
    def trivial():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
                    t += 1
        return {"U": U, "V": V, "W": W}

    # ---- Strassen base for 2x2 x 2x2 -> 2x2 (rank 7) ----
    def strassen222():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[3][4] = -1
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = -1
        return U, V, W

    # ---- Verification: reconstruct multiplication tensor and compare ----
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        return True

    # ---- Correct block decomposition using Kronecker product structure ----
    def block_decomp():
        sU, sV, sW = strassen222()
        # 4x4 x 4x4 part: rank 7*7 = 49
        # 4x4 x 4x3 part: rank 4*4*3 = 48
        total_rank = 49 + 48
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        # For the 4x4 x 4x4 part, we use 7x7 Kronecker structure:
        # A is 4x4, split into 2x2 blocks of 2x2 submatrices
        # A[br*2+lr, bc*2+lc] for block (br,bc), local (lr,lc)
        # Similarly for B and C.
        # The 2x2x2 Strassen gives 7 terms for one 2x2x2 block.
        # For 4x4x4, we need 7x7 = 49 terms (Kronecker product of two 2x2x2 Strassen).
        
        # The 4x4 matrix A has blocks A00, A01, A10, A11 (each 2x2)
        # The 4x4 matrix B (first 4 cols) has blocks B00, B01, B10, B11 (each 2x2)
        # C = A * B_first4 has blocks C00, C01, C10, C11 (each 2x2)
        
        # For each pair of Strassen steps (s1, s2) from the 2x2x2 algorithm,
        # we combine them:
        # U_term = sU[s1] ⊗ sU[s2]  (Kronecker product, giving 4x4 structure)
        # V_term = sV[s1] ⊗ sV[s2]
        # W_term = sW[s1] ⊗ sW[s2]
        
        # But we need to be careful: the Kronecker product structure means
        # that for step (s1, s2), the U factor at position (i,j) is:
        # U[i,j] = sU[i//2][s1] * sU[i%2 * 2 + j//2][s2] ... no that's not right.
        
        # Actually, let's think differently. The 4x4x4 multiplication tensor can be
        # decomposed as the tensor product of two 2x2x2 multiplication tensors.
        # If T_2 is the 2x2x2 multiplication tensor (rank 7 via Strassen),
        # then T_4 = T_2 ⊗ T_2 has rank 7*7 = 49.
        
        # The tensor T_2 has indices (i, j, k) for i,j,k in {0,1}
        # T_4 has indices (I, J, K) where I = i*2 + i', J = j*2 + j', K = k*2 + k'
        # T_4[I, J, K] = T_2[i, j, k] * T_2[i', j', k']
        
        # For the decomposition:
        # sum_t U[I, J, t] * V[J, K, t] * W[K, I, t] = T_4[I, J, K]
        
        # If we use t = (t1, t2) where t1, t2 in {0..6}:
        # U[I, J, (t1,t2)] = sU[i, t1] * sU[i', t2]  where I = i*2+i', J = j*2+j'
        # V[J, K, (t1,t2)] = sV[j, t1] * sV[j', t2]  where J = j*2+j', K = k*2+k'
        # W[K, I, (t1,t2)] = sW[k, t1] * sW[k', t2]  where K = k*2+k', I = i*2+i'
        
        # Let's verify: sum_{t1,t2} sU[i,t1]*sU[i',t2] * sV[j,t1]*sV[j',t2] * sW[k,t1]*sW[k',t2]
        # = (sum_{t1} sU[i,t1]*sV[j,t1]*sW[k,t1]) * (sum_{t2} sU[i',t2]*sV[j',t2]*sW[k',t2])
        # = T_2[i,j,k] * T_2[i',j',k'] = T_4[I,J,K] ✓
        
        # Now map to the actual index conventions:
        # U index: i*m + j where i is row of A (4 values), j is col of A / row of B (4 values)
        #   In our 2x2 block notation: i = bi*2 + ii, j = bj*2 + jj
        # V index: j*p + k where j is row of B (4 values for first 4 cols), k is col of B (4 values)
        #   j = bj*2 + jj, k = bk*2 + kk
        # W index: k*n + i where k is col of C (4 values), i is row of C (4 values)
        #   k = bk*2 + kk, i = bi*2 + ii
        
        # So:
        # U[bi*2+ii, bj*2+jj, t1*7+t2] = sU[ii*2+jj, t1] * sU[ii*2+jj, t2]
        #   Wait, that's not right either. Let me re-examine.
        
        # Actually sU has shape (4, 7) where the 4 indices are (i, j) for 2x2 matrix A.
        # sU[i*2+j, t] is the U factor for row i, col j, term t.
        
        # For the 4x4 case:
        # U[i*4+j, t1*7+t2] where i = bi*2+ii, j = bj*2+jj
        # We need: U[i*4+j, t1*7+t2] = sU[ii*2+jj, t1] * sU[ii*2+jj, t2]
        #   NO! That's wrong. The Kronecker product means:
        #   U[i*4+j, (t1,t2)] = sU[bi, bj, t1] * sU[ii, jj, t2]
        #   But sU is for 2x2, so sU has indices (i,j) where i,j in {0,1}
        #   sU[i*2+j, t]
        
        # Let me redo: For 2x2x2, U[i*2+j, t] for i,j in {0,1}
        # For 4x4x4 via Kronecker:
        #   U[(bi*2+ii)*4 + (bj*2+jj), t1*7+t2] = sU[bi*2+bj, t1] * sU[ii*2+jj, t2]
        
        # Similarly:
        #   V[(bj*2+jj)*4 + (bk*2+kk), t1*7+t2] = sV[bj*2+bk, t1] * sV[jj*2+kk, t2]
        
        # And W:
        #   W[(bk*2+kk)*4 + (bi*2+ii), t1*7+t2] = sW[bk*2+bi, t1] * sW[kk*2+ii, t2]
        
        # Let's build this:
        for bi in range(2):
            for bj in range(2):
                for ii in range(2):
                    for jj in range(2):
                        I = bi * 2 + ii
                        J = bj * 2 + jj
                        U_idx = I * m + J  # I*4 + J
                        for t1 in range(7):
                            for t2 in range(7):
                                t = t1 * 7 + t2
                                U[U_idx][t] += sU[bi * 2 + bj][t1] * sU[ii * 2 + jj][t2]
        
        for bj in range(2):
            for jj in range(2):
                for bk in range(2):
                    for kk in range(2):
                        J = bj * 2 + jj
                        K = bk * 2 + kk
                        V_idx = J * 4 + K  # J*4 + K, for the first 4 columns
                        for t1 in range(7):
                            for t2 in range(7):
                                t = t1 * 7 + t2
                                V[V_idx][t] += sV[bj * 2 + bk][t1] * sV[jj * 2 + kk][t2]
        
        for bk in range(2):
            for kk in range(2):
                for bi in range(2):
                    for ii in range(2):
                        K = bk * 2 + kk
                        I = bi * 2 + ii
                        W_idx = K * n + I  # K*4 + I
                        for t1 in range(7):
                            for t2 in range(7):
                                t = t1 * 7 + t2
                                W[W_idx][t] += sW[bk * 2 + bi][t1] * sW[kk * 2 + ii][t2]
        
        # ---- Schoolbook for columns 4..6 (4x4 x 4x3) ----
        base = 49
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(4, 7):
                    U[i * m + j][base + t] = 1
                    V[j * p + k][base + t] = 1
                    W[k * n + i][base + t] = 1
                    t += 1

        return {"U": U, "V": V, "W": W}, total_rank

    # ---- Try the structural decomposition, verify, else fall back ----
    try:
        if time_left() > 2.0:
            cand, rank = block_decomp()
            if verify(cand["U"], cand["V"], cand["W"], rank):
                return cand
    except Exception:
        pass

    # Safety fallback: guaranteed valid trivial decomposition
    return trivial()
