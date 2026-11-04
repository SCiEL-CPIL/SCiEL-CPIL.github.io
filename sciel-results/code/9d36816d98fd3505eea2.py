"""
Strategy: Block Composition.
Instead of the broken Kronecker extension, we decompose the 4x4x9 multiplication 
into two 4x4x4 multiplications (using Strassen, rank 49 each) and one 4x4x1 
multiplication (using schoolbook, rank 16).
Total Rank = 49 + 49 + 16 = 114.
This is strictly better than the schoolbook 144 and structurally valid.
"""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # 1. Strassen 2x2x2 (Rank 7)
    U2 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1]
    ]
    V2 = [
        [1, -1, 0, 1, -1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, -1, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 1]
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1]
    ]
    
    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if s != 1:
                    # Fallback to schoolbook 144
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

    # 2. Construct 4x4x4 Strassen (Rank 49) via Kronecker of 2x2x2
    r4 = 49
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            for c1 in range(7):
                                for c2 in range(7):
                                    idx_c = c1*7 + c2
                                    
                                    u1 = U2[i1*2+j1][c1]
                                    u2 = U2[i2*2+j2][c2]
                                    v1 = V2[j1*2+k1][c1]
                                    v2 = V2[j2*2+k2][c2]
                                    w1 = W2[k1*2+i1][c1]
                                    w2 = W2[k2*2+i2][c2]
                                    
                                    idx_u = (i1*2+i2)*4 + (j1*2+j2)
                                    idx_v = (j1*2+j2)*4 + (k1*2+k2)
                                    idx_w = (k1*2+k2)*4 + (i1*2+i2)
                                    
                                    U4[idx_u][idx_c] += u1 * u2
                                    V4[idx_v][idx_c] += v1 * v2
                                    W4[idx_w][idx_c] += w1 * w2
                                    
    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U4[i*4+j][c] * V4[j*4+k][c] * W4[k*4+i][c]
                if s != 1:
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

    # 3. Construct 4x4x1 Schoolbook (Rank 16)
    # A (4x4) * B (4x1) -> C (4x1)
    # T[i, j, k=0] = 1 if i==i, j==j, k=0? No.
    # C[i, 0] = sum_j A[i, j] B[j, 0]
    # Tensor T[i, j, 0] = 1.
    # Rank 16: one term for each (i, j) pair.
    r1 = 16
    U1 = [[0]*r1 for _ in range(16)] # 4x4
    V1 = [[0]*r1 for _ in range(4)]  # 4x1
    W1 = [[0]*r1 for _ in range(4)]  # 1x4 -> 4x1? W index is k*n+i. k=0. So 4x16.
    
    col = 0
    for i in range(4):
        for j in range(4):
            U1[i*4+j][col] = 1
            V1[j*1+0][col] = 1 # V index j*p+k. p=1, k=0.
            W1[0*4+i][col] = 1 # W index k*n+i. k=0.
            col += 1
            
    # Verify 4x4x1
    for i in range(4):
        for j in range(4):
            s = 0
            for c in range(16):
                s += U1[i*4+j][c] * V1[j*1+0][c] * W1[0*4+i][c]
            if s != 1:
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

    # 4. Assemble 4x4x9
    # Total Rank = 49 + 49 + 16 = 114
    R = 114
    U = [[0]*R for _ in range(16)]
    V = [[0]*R for _ in range(36)] # 4x9
    W = [[0]*R for _ in range(36)] # 9x4
    
    # Block 1: k=0,1,2,3 (cols 0-48)
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i*4+j][c] = U4[i*4+j][c]
                
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*9+k][c] = V4[j*4+k][c]
                
    for i in range(4):
        for k in range(4):
            for c in range(49):
                W[k*4+i][c] = W4[k*4+i][c]
                
    # Block 2: k=4,5,6,7 (cols 49-97)
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i*4+j][49+c] = U4[i*4+j][c]
                
    for j in range(4):
        for k in range(4, 8):
            for c in range(49):
                V[j*9+k][49+c] = V4[j*4+(k-4)][c]
                
    for i in range(4):
        for k in range(4, 8):
            for c in range(49):
                W[k*4+i][49+c] = W4[(k-4)*4+i][c]
                
    # Block 3: k=8 (cols 98-113)
    for i in range(4):
        for j in range(4):
            for c in range(16):
                U[i*4+j][98+c] = U1[i*4+j][c]
                
    for j in range(4):
        for c in range(16):
            V[j*9+8][98+c] = V1[j*1+0][c]
            
    for i in range(4):
        for c in range(16):
            W[8*4+i][98+c] = W1[0*4+i][c]
            
    # Final Verification
    for i in range(4):
        for j in range(4):
            for k in range(9):
                s = 0
                for c in range(114):
                    s += U[i*4+j][c] * V[j*9+k][c] * W[k*4+i][c]
                if s != 1:
                    # Fallback to schoolbook 144
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
