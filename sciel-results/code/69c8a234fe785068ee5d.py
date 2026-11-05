import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 2, 4, 5
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Try rank-32 solution with 2+3 split on p and 2+2 split on m
    # Sub-problems:
    #   (i in {0,1}, j in {0,1}, k in {0,1}): 2x2x2, Strassen rank 7
    #   (i in {0,1}, j in {0,1}, k in {2,3,4}): 2x2x3, rank 9
    #   (i in {0,1}, j in {2,3}, k in {0,1}): 2x2x2, Strassen rank 7
    #   (i in {0,1}, j in {2,3}, k in {2,3,4}): 2x2x3, rank 9
    # Total: 7+9+7+9 = 32
    
    rank32 = 32
    U32 = [[0] * rank32 for _ in range(n * m)]
    V32 = [[0] * rank32 for _ in range(m * p)]
    W32 = [[0] * rank32 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        """Add Strassen 2x2x2 block"""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (A11+A22)(B11+B22)
        c = col_offset + 0
        U32[a11][c] += 1
        U32[a22][c] += 1
        V32[b11][c] += 1
        V32[b22][c] += 1
        W32[c11][c] += 1
        W32[c22][c] += 1
        
        # M2 = (A21+A22)B11
        c = col_offset + 1
        U32[a21][c] += 1
        U32[a22][c] += 1
        V32[b11][c] += 1
        W32[c21][c] += 1
        W32[c22][c] += -1
        
        # M3 = A11(B12-B22)
        c = col_offset + 2
        U32[a11][c] += 1
        V32[b12][c] += 1
        V32[b22][c] += -1
        W32[c12][c] += 1
        W32[c22][c] += 1
        
        # M4 = A22(B21-B11)
        c = col_offset + 3
        U32[a22][c] += 1
        V32[b21][c] += 1
        V32[b11][c] += -1
        W32[c11][c] += 1
        W32[c21][c] += 1
        
        # M5 = (A11+A12)B22
        c = col_offset + 4
        U32[a11][c] += 1
        U32[a12][c] += 1
        V32[b22][c] += 1
        W32[c11][c] += -1
        W32[c12][c] += 1
        
        # M6 = (A21-A11)(B11+B12)
        c = col_offset + 5
        U32[a21][c] += 1
        U32[a11][c] += -1
        V32[b11][c] += 1
        V32[b12][c] += 1
        W32[c22][c] += 1
        
        # M7 = (A12-A22)(B21+B22)
        c = col_offset + 6
        U32[a12][c] += 1
        U32[a22][c] += -1
        V32[b21][c] += 1
        V32[b22][c] += 1
        W32[c11][c] += 1
    
    def add_223(i1, i2, j1, j2, k1, k2, k3, col_offset):
        """Add rank-9 decomposition for 2x2x3 block"""
        # A is 2x2 (rows i1,i2; cols j1,j2)
        # B is 2x3 (rows j1,j2; cols k1,k2,k3)
        # C is 2x3 (rows i1,i2; cols k1,k2,k3)
        
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b13 = j1 * p + k3
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        b23 = j2 * p + k3
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c13 = k3 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        c23 = k3 * n + i2
        
        # Use Strassen for columns k1,k2 (first 7 terms)
        # M1 = (A11+A22)(B11+B22)
        c = col_offset + 0
        U32[a11][c] += 1
        U32[a22][c] += 1
        V32[b11][c] += 1
        V32[b22][c] += 1
        W32[c11][c] += 1
        W32[c22][c] += 1
        
        # M2 = (A21+A22)B11
        c = col_offset + 1
        U32[a21][c] += 1
        U32[a22][c] += 1
        V32[b11][c] += 1
        W32[c21][c] += 1
        W32[c22][c] += -1
        
        # M3 = A11(B12-B22)
        c = col_offset + 2
        U32[a11][c] += 1
        V32[b12][c] += 1
        V32[b22][c] += -1
        W32[c12][c] += 1
        W32[c22][c] += 1
        
        # M4 = A22(B21-B11)
        c = col_offset + 3
        U32[a22][c] += 1
        V32[b21][c] += 1
        V32[b11][c] += -1
        W32[c11][c] += 1
        W32[c21][c] += 1
        
        # M5 = (A11+A12)B22
        c = col_offset + 4
        U32[a11][c] += 1
        U32[a12][c] += 1
        V32[b22][c] += 1
        W32[c11][c] += -1
        W32[c12][c] += 1
        
        # M6 = (A21-A11)(B11+B12)
        c = col_offset + 5
        U32[a21][c] += 1
        U32[a11][c] += -1
        V32[b11][c] += 1
        V32[b12][c] += 1
        W32[c22][c] += 1
        
        # M7 = (A12-A22)(B21+B22)
        c = col_offset + 6
        U32[a12][c] += 1
        U32[a22][c] += -1
        V32[b21][c] += 1
        V32[b22][c] += 1
        W32[c11][c] += 1
        
        # Now handle column k3 with 2 additional terms
        # C[0,k3] = A[0,j1]*B[j1,k3] + A[0,j2]*B[j2,k3]
        # C[1,k3] = A[1,j1]*B[j1,k3] + A[1,j2]*B[j2,k3]
        
        # M8 = (A11+A22) * (B13+B23)
        # Contributes: C[0,k3] += 1, C[1,k3] += 1 (but this is wrong)
        # Actually: (A11+A22)(B13+B23) = A11*B13 + A11*B23 + A22*B13 + A22*B23
        # This doesn't directly give what we want.
        
        # Let's use a different approach for the third column.
        # We need:
        # C[0,k3] = A11*B13 + A12*B23
        # C[1,k3] = A21*B13 + A22*B23
        
        # Use 2 terms:
        # M8 = (A11+A21) * B13
        # M9 = (A12+A22) * B23
        
        # M8: U: a11=1, a21=1; V: b13=1; W: c21=1 (C[1,k3] gets A21*B13)
        # But we also need C[0,k3] to get A11*B13. So W: c11=1, c21=1
        # Wait, that's not right either. Let me think again.
        
        # Actually, for the third column, we can use:
        # M8 = A11 * B13 -> C[0,k3]
        # M9 = A21 * B13 -> C[1,k3]
        # M10 = A12 * B23 -> C[0,k3]
        # M11 = A22 * B23 -> C[1,k3]
        # That's 4 terms, giving total 7+4=11, not 9.
        
        # To get rank 9, we need a smarter decomposition.
        # Known result: rank of 2x2x3 is 9.
        # One decomposition:
        # Use 7 Strassen terms for cols 0,1, and for col 2 use:
        # M8 = (A11 + A12) * B23 -> contributes to C[0,k3] and C[1,k3]? No.
        
        # Let me just use 4 terms for the third column, giving rank 11 per 2x2x3 block.
        # Total: 7+11+7+11 = 36, same as before.
        
        # For now, let's just do schoolbook for the third column (4 terms)
        # M8 = A11 * B13
        c = col_offset + 7
        U32[a11][c] += 1
        V32[b13][c] += 1
        W32[c13][c] += 1
        
        # M9 = A21 * B13
        c = col_offset + 8
        U32[a21][c] += 1
        V32[b13][c] += 1
        W32[c21][c] += 1
        
        # M10 = A12 * B23
        c = col_offset + 9
        U32[a12][c] += 1
        V32[b23][c] += 1
        W32[c13][c] += 1
        
        # M11 = A22 * B23
        c = col_offset + 10
        U32[a22][c] += 1
        V32[b23][c] += 1
        W32[c23][c] += 1
        
        # This uses 11 columns, not 9. So total would be 7+11+7+11=36.
        # For a true rank-32 solution, we need a rank-9 decomposition of 2x2x3.
        # Let's fall back to rank 36 for now.
        pass
    
    # Since finding a verified rank-9 decomposition for 2x2x3 is complex,
    # let's stick with the proven rank-36 solution from the current program.
    
    rank36 = 36
    U36 = [[0] * rank36 for _ in range(n * m)]
    V36 = [[0] * rank36 for _ in range(m * p)]
    W36 = [[0] * rank36 for _ in range(p * n)]
    
    col = 0
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    standard_terms = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({2:1}, {0:1}, {1:1}),
        ({3:1}, {1:1}, {1:1})
    ]
    
    for q in range(2):
        for s in range(3):
            if s < 2:
                for u_t, v_t, w_t in strassen_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v // 2
                        lk = idx_v % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w // 2
                        lk = idx_w % 2
                        gi = li
                        gk = 2*s + lk
                        W36[gk*n + gi][col] += val_w
                    col += 1
            else:
                for u_t, v_t, w_t in standard_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v
                        gj = 2*q + lj
                        gk = 4
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w
                        gi = li
                        gk = 4
                        W36[gk*n + gi][col] += val_w
                    col += 1
    
    if col == rank36 and verify(U36, V36, W36, rank36):
        return {"U": U36, "V": V36, "W": W36}
    
    # Schoolbook fallback
    rank_school = n * m * p
    best = {
        "U": [[0]*rank_school for _ in range(n*m)],
        "V": [[0]*rank_school for _ in range(m*p)],
        "W": [[0]*rank_school for _ in range(p*n)]
    }
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best["U"][i*m+j][c] = 1
                best["V"][j*p+k][c] = 1
                best["W"][k*n+i][c] = 1
                c += 1
    return best
