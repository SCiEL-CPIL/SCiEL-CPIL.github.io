"""Matrix multiplication 4,4,9 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 9

    # 1. Define Strassen 2x2x2 decomposition (Rank 7)
    # Indices: U (A indices i,j), V (B indices j,k), W (C indices k,i)
    # Convention: C_{ki} = sum_{j,l} A_{ij} B_{jl}
    # Strassen terms M1..M7:
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = A11(B10-B00)       -> C10
    # M3 = (A10+A11)B01       -> C11 (wait, standard Strassen: M3 = (A10+A11)B01 -> C01? No.)
    # Let's use the standard explicit formulas for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[q, r], [s, t]]
    # M1 = (a+d)(e+h)
    # M2 = d(f-h)
    # M3 = (c+d)g
    # M4 = a(e-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # q = M1 + M4 - M5 + M7
    # r = M2 + M5
    # s = M3 + M6
    # t = M1 - M2 + M3 - M6
    
    # Mapping to 1D indices:
    # A: 00->0, 01->1, 10->2, 11->3
    # B: 00->0, 01->1, 10->2, 11->3
    # C: 00->0 (k=0,i=0), 01->1 (k=0,i=1), 10->2 (k=1,i=0), 11->3 (k=1,i=1)
    # Note: W index is k*n + i. For 2x2, n=2.
    # W[0] is C_00 (k=0, i=0)
    # W[1] is C_01 (k=0, i=1)
    # W[2] is C_10 (k=1, i=0)
    # W[3] is C_11 (k=1, i=1)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # U: A00(0), A11(3)
    U2[0][0] = 1; U2[3][0] = 1
    # V: B00(0), B11(3)
    V2[0][0] = 1; V2[3][0] = 1
    # W: C00(0) += 1, C11(3) += 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2: A11(B10-B00)
    # U: A11(3)
    U2[3][1] = 1
    # V: B10(2), B00(0)
    V2[2][1] = 1; V2[0][1] = -1
    # W: C10(2) += 1
    W2[2][1] = 1
    
    # M3: (A10+A11)B01
    # U: A10(2), A11(3)
    U2[2][2] = 1; U2[3][2] = 1
    # V: B01(1)
    V2[1][2] = 1
    # W: C11(3) += 1 ? 
    # Wait, standard Strassen: s = M3 + M6. s is C10? 
    # Let's re-verify standard Strassen output mapping.
    # q (C00) = M1 + M4 - M5 + M7
    # r (C01) = M2 + M5
    # s (C10) = M3 + M6
    # t (C11) = M1 - M2 + M3 - M6
    # 
    # So:
    # M3 contributes to C10 (index 2) and C11 (index 3).
    # My previous mapping for M3 was W2[2][2]=1 (C10). I need to add W2[3][2]=1 (C11).
    W2[2][2] = 1; W2[3][2] = 1
    
    # M4: A00(B01-B11)
    # U: A00(0)
    U2[0][3] = 1
    # V: B01(1), B11(3)
    V2[1][3] = 1; V2[3][3] = -1
    # W: C00(0) += 1
    W2[0][3] = 1
    
    # M5: (A00+A01)B11
    # U: A00(0), A01(1)
    U2[0][4] = 1; U2[1][4] = 1
    # V: B11(3)
    V2[3][4] = 1
    # W: C00(0) -= 1, C01(1) += 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6: (A10-A00)(B00+B10)
    # U: A10(2), A00(0)
    U2[2][5] = 1; U2[0][5] = -1
    # V: B00(0), B10(2)
    V2[0][5] = 1; V2[2][5] = 1
    # W: C10(2) += 1, C11(3) -= 1
    W2[2][5] = 1; W2[3][5] = -1
    
    # M7: (A01-A11)(B01+B11)
    # U: A01(1), A11(3)
    U2[1][6] = 1; U2[3][6] = -1
    # V: B01(1), B11(3)
    V2[1][6] = 1; V2[3][6] = 1
    # W: C00(0) += 1, C01(1) += 1
    W2[0][6] = 1; W2[1][6] = 1

    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                expected = 1
                if s != expected:
                    # Fallback to schoolbook if Strassen definition is wrong
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    # 2. Kronecker Product to get 4x4x4 (Rank 49)
    r4 = 49
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]
    
    # Indices for 4x4:
    # A: (i1,j1,i2,j2) -> (i1*2+i2)*4 + (j1*2+j2)
    # B: (j1,k1,j2,k2) -> (j1*2+j2)*4 + (k1*2+k2)
    # C: (k1,i1,k2,i2) -> (k1*2+k2)*4 + (i1*2+i2)
    
    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(7):
                u1 = U2[i1*2+j1][c1]
                if u1 == 0: continue
                for i2 in range(2):
                    for j2 in range(2):
                        for c2 in range(7):
                            u2 = U2[i2*2+j2][c2]
                            if u2 == 0: continue
                            idx_u = (i1*2 + i2)*4 + (j1*2 + j2)
                            idx_c = c1*7 + c2
                            U4[idx_u][idx_c] = u1 * u2

    for j1 in range(2):
        for k1 in range(2):
            for c1 in range(7):
                v1 = V2[j1*2+k1][c1]
                if v1 == 0: continue
                for j2 in range(2):
                    for k2 in range(2):
                        for c2 in range(7):
                            v2 = V2[j2*2+k2][c2]
                            if v2 == 0: continue
                            idx_v = (j1*2 + j2)*4 + (k1*2 + k2)
                            idx_c = c1*7 + c2
                            V4[idx_v][idx_c] = v1 * v2

    for k1 in range(2):
        for i1 in range(2):
            for c1 in range(7):
                w1 = W2[k1*2+i1][c1]
                if w1 == 0: continue
                for k2 in range(2):
                    for i2 in range(2):
                        for c2 in range(7):
                            w2 = W2[k2*2+i2][c2]
                            if w2 == 0: continue
                            idx_w = (k1*2 + k2)*4 + (i1*2 + i2)
                            idx_c = c1*7 + c2
                            W4[idx_w][idx_c] = w1 * w2

    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U4[i*4+j][c] * V4[j*4+k][c] * W4[k*4+i][c]
                if s != 1:
                    # Fallback
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    # 3. Extend to 4x4x9
    # Strategy: Split output columns k into 0..3 (handled by U4/V4/W4) and 4..8 (handled by schoolbook)
    # Total Rank = 49 (Strassen 4x4x4) + 4*1*5 (Schoolbook 4x4x5) = 49 + 20 = 69
    
    rank_total = 49 + 20
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x 69
    V = [[0] * rank_total for _ in range(m * p)]  # 36 x 69
    W = [[0] * rank_total for _ in range(p * n)]  # 36 x 69
    
    # Part 1: Strassen 4x4x4 (columns 0-3)
    # U is same as U4 (padded with zeros for the rest)
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            
    # V: For j in 0..3, k in 0..3, use V4[j*4+k]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*9+k][c] = V4[j*4+k][c]
                
    # W: For k in 0..3, i in 0..3, use W4[k*4+i]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4+i][c] = W4[k*4+i][c]

    # Part 2: Schoolbook for columns 4..8
    # Columns 4, 5, 6, 7, 8
    # Rank offset = 49
    for k in range(4, 9):
        for i in range(4):
            for j in range(4):
                c = 49 + (k-4)*16 + i*4 + j
                U[i*4+j][c] = 1
                V[j*9+k][c] = 1
                W[k*4+i][c] = 1

    # Final Verification
    for i in range(4):
        for j in range(4):
            for k in range(9):
                s = 0
                for c in range(rank_total):
                    s += U[i*4+j][c] * V[j*9+k][c] * W[k*4+i][c]
                if s != 1:
                    # Fallback to full schoolbook
                    rank = n * m * p
                    U_f = [[0] * rank for _ in range(n * m)]
                    V_f = [[0] * rank for _ in range(m * p)]
                    W_f = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_f[ii*m+jj][col] = 1
                                V_f[jj*p+kk][col] = 1
                                W_f[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
