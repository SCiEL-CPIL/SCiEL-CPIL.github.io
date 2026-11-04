"""Improved program using Hybrid Block Decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Hybrid Decomposition
    # Split the 9 columns of B into 4 and 5.
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4) using Strassen (Rank 49)
    # Part 2: A (4x4) * B2 (4x5) -> C2 (4x5) using Schoolbook (Rank 80)
    # Total Rank: 49 + 80 = 129
    
    # --- Part 1: Strassen 4x4x4 ---
    # We reuse the Strassen 2x2x2 logic to build 4x4x4.
    
    # Strassen 2x2x2
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # M2
    U2[3][1] = 1
    V2[2][1] = 1; V2[0][1] = -1
    W2[1][1] = 1
    # M3
    U2[2][2] = 1; U2[3][2] = 1
    V2[1][2] = 1
    W2[2][2] = 1
    # M4
    U2[0][3] = 1
    V2[1][3] = 1; V2[3][3] = -1
    W2[0][3] = 1
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = 1
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[0][5] = 1; W2[1][5] = 1
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[1][6] = 1; V2[3][6] = 1
    W2[2][6] = 1; W2[3][6] = 1

    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
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

    # Build 4x4x4 via Kronecker
    r4 = 49
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]
    
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

    # --- Part 2: Schoolbook 4x4x5 ---
    # Rank = 4 * 4 * 5 = 80
    r5 = 80
    U5 = [[0]*r5 for _ in range(16)]
    V5 = [[0]*r5 for _ in range(20)] # 4 rows, 5 cols
    W5 = [[0]*r5 for _ in range(20)] # 5 rows, 4 cols
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U5[i*4+j][col] = 1
                V5[j*5+k][col] = 1
                W5[k*4+i][col] = 1
                col += 1

    # Verify 4x4x5
    for i in range(4):
        for j in range(4):
            for k in range(5):
                s = 0
                for c in range(80):
                    s += U5[i*4+j][c] * V5[j*5+k][c] * W5[k*4+i][c]
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

    # --- Combine ---
    # Total Rank = 49 + 80 = 129
    total_rank = 129
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)] # 4x9
    W = [[0]*total_rank for _ in range(36)] # 9x4
    
    # Fill Part 1 (First 49 columns of rank)
    # U: 16x49 -> U[0:16, 0:49]
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            
    # V: 36x49 -> V[0:16, 0:49] corresponds to B1 (4x4)
    # B1 is cols 0-3 of B.
    # V4 is 16x49 (j*4+k).
    # V is 36x49 (j*9+k).
    # For j in 0..3, k in 0..3: V[j*9+k] = V4[j*4+k]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*9+k][c] = V4[j*4+k][c]
                
    # W: 36x49 -> W[0:16, 0:49] corresponds to C1 (4x4)
    # W4 is 16x49 (k*4+i).
    # W is 36x49 (k*4+i) for k in 0..3.
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4+i][c] = W4[k*4+i][c]

    # Fill Part 2 (Next 80 columns of rank)
    offset = 49
    # U: 16x80 -> U[0:16, 49:129]
    for i in range(16):
        for c in range(80):
            U[i][offset + c] = U5[i][c]
            
    # V: 36x80 -> V[0:20, 49:129] corresponds to B2 (4x5)
    # B2 is cols 4-8 of B.
    # V5 is 20x80 (j*5+k).
    # V is 36x80 (j*9+k).
    # For j in 0..3, k in 0..4: V[j*9 + (4+k)] = V5[j*5+k]
    for j in range(4):
        for k in range(5):
            for c in range(80):
                V[j*9 + (4+k)][offset + c] = V5[j*5+k][c]
                
    # W: 36x80 -> W[16:36, 49:129] corresponds to C2 (4x5)
    # C2 is cols 4-8 of C.
    # W5 is 20x80 (k*4+i) where k is 0..4.
    # W is 36x80 (k*4+i) where k is 4..8.
    # For k in 0..4, i in 0..3: W[(4+k)*4 + i] = W5[k*4+i]
    for k in range(5):
        for i in range(4):
            for c in range(80):
                W[(4+k)*4 + i][offset + c] = W5[k*4+i][c]

    # Final Verification
    for i in range(4):
        for j in range(4):
            for k in range(9):
                s = 0
                for c in range(129):
                    s += U[i*4+j][c] * V[j*9+k][c] * W[k*4+i][c]
                if s != 1:
                    # Fallback to schoolbook
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
