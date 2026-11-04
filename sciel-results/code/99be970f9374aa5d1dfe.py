"""Matrix multiplication 4x4x6 - Improved Strassen-based rank reduction."""
import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p  # 96
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # 2x2x2 Strassen factors
    # A, B are 2x2. C = A*B.
    # U (4,7), V (4,7), W (4,7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen coefficients
    # s0: (A00+A11)(B00+B11)
    # s1: (A10+A11)B00
    # s2: A00(B01-B11)
    # s3: A11(B10-B00)
    # s4: (A00+A01)B11
    # s5: (A10-A00)(B10+B11)
    # s6: (A01-A11)(B00+B10)
    
    # alpha_s[block]
    alpha = [
        [1, 0, 0, 1],  # s0
        [0, 0, 1, 1],  # s1
        [1, 0, 0, 0],  # s2
        [0, 0, 0, 1],  # s3
        [1, 1, 0, 0],  # s4
        [-1, 0, 1, 0], # s5
        [0, 1, 0, -1]  # s6
    ]
    # beta_s[block]
    beta = [
        [1, 0, 0, 1],  # s0
        [1, 0, 0, 0],  # s1
        [0, 1, 0, -1], # s2
        [-1, 0, 1, 0], # s3
        [0, 0, 0, 1],  # s4
        [1, 1, 0, 0],  # s5
        [0, 0, 1, 1]   # s6
    ]
    # gamma_s[out_block]
    # C00 = s0 - s1 - s3 + s4 + s6
    # C01 = s2 + s4
    # C10 = s1 + s3
    # C11 = s0 - s2 + s3 - s5
    gamma = [
        [1, -1, 0, 0, 1, 0, 1],  # s0..s6 coeff for C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, -1, 1, 0, -1, 0]  # C11
    ]
    
    for s in range(7):
        # U2: A part
        for a in range(4):
            if alpha[s][a] != 0:
                ar, br = a // 2, a % 2
                U2[ar * 2 + br, s] = alpha[s][a]
        
        # V2: B part
        for b in range(4):
            if beta[s][b] != 0:
                ar, br = b // 2, b % 2
                V2[ar * 2 + br, s] = beta[s][b]
        
        # W2: C part
        for c in range(4):
            if gamma[c][s] != 0:
                ar, br = c // 2, c % 2
                W2[ar * 2 + br, s] = gamma[c][s]
    
    # Build 4x4x4 factors from 2x2x2
    # U4 (16, 28), V4 (16, 28), W4 (16, 28)
    U4 = np.zeros((16, 28), dtype=int)
    V4 = np.zeros((16, 28), dtype=int)
    W4 = np.zeros((16, 28), dtype=int)
    
    for s in range(7):
        for a in range(4):
            if alpha[s][a] != 0:
                ar, br = a // 2, a % 2
                # A block (ar, br) is 2x2.
                # Local indices in U2: r, c
                # Global indices in U4: (ar*2+r)*4 + (br*2+c)
                for r in range(2):
                    for c in range(2):
                        if U2[r*2+c, s] != 0:
                            U4[(ar*2+r)*4 + (br*2+c), 7*s + s] += U2[r*2+c, s] * alpha[s][a]
        
        for b in range(4):
            if beta[s][b] != 0:
                ar, br = b // 2, b % 2
                for r in range(2):
                    for c in range(2):
                        if V2[r*2+c, s] != 0:
                            V4[(ar*2+r)*4 + (br*2+c), 7*s + s] += V2[r*2+c, s] * beta[s][b]
        
        for c_out in range(4):
            if gamma[c_out][s] != 0:
                ar, br = c_out // 2, c_out % 2
                for r in range(2):
                    for c in range(2):
                        if W2[r*2+c, s] != 0:
                            W4[(ar*2+r)*4 + (br*2+c), 7*s + s] += W2[r*2+c, s] * gamma[c_out][s]
    
    # Verify 4x4x4
    def verify_4x4x4(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s_val = 0
                    for r in range(28):
                        s_val += U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                    if s_val != 1:
                        return False
        return True
    
    if not verify_4x4x4(U4, V4, W4):
        return schoolbook()
    
    # Build 4x4x2 factors from 4x4x4
    # U2_4 (16, 28) is same as U4
    # V2_4 (8, 28) : V4 rows for k=0,1
    # W2_4 (8, 28) : W4 rows for k=0,1
    
    U_4x4x2 = U4
    V_4x4x2 = np.zeros((8, 28), dtype=int)
    W_4x4x2 = np.zeros((8, 28), dtype=int)
    
    for j in range(4):
        for k in range(2):
            V_4x4x2[j*2+k, :] = V4[j*4+k, :]
            
    for k in range(2):
        for i in range(4):
            W_4x4x2[k*4+i, :] = W4[k*4+i, :]
    
    # Verify 4x4x2
    def verify_4x4x2(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    s_val = 0
                    for r in range(28):
                        s_val += U[i*4+j, r] * V[j*2+k, r] * W[k*4+i, r]
                    if s_val != 1:
                        return False
        return True
    
    if not verify_4x4x2(U_4x4x2, V_4x4x2, W_4x4x2):
        return schoolbook()
        
    # Assemble 4x4x6
    # Part 1: 4x4x4 (cols 0-3)
    # Part 2: 4x4x2 (cols 4-5)
    
    R = 56
    U_final = np.zeros((16, R), dtype=int)
    V_final = np.zeros((24, R), dtype=int)
    W_final = np.zeros((24, R), dtype=int)
    
    # Copy Part 1
    U_final[:, 0:28] = U4
    V_final[0:16, 0:28] = V4
    W_final[0:16, 0:28] = W4
    
    # Copy Part 2
    U_final[:, 28:56] = U_4x4x2
    # V_4x4x2 (8, 28) -> V_final (16:24, 28:56)
    # V_4x4x2 index j*2+k -> V_final index j*6 + (4+k)
    for j in range(4):
        for k in range(2):
            V_final[j*6 + 4 + k, 28:56] = V_4x4x2[j*2+k, :]
            
    # W_4x4x2 (8, 28) -> W_final (16:24, 28:56)
    # W_4x4x2 index k*4+i -> W_final index (4+k)*4 + i
    for k in range(2):
        for i in range(4):
            W_final[(4+k)*4 + i, 28:56] = W_4x4x2[k*4+i, :]
            
    # Verify 4x4x6
    def verify_4x4x6(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    s_val = 0
                    for r in range(56):
                        s_val += U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r]
                    if s_val != 1:
                        return False
        return True
    
    if not verify_4x4x6(U_final, V_final, W_final):
        return schoolbook()
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
