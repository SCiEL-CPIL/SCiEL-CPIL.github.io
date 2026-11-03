import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    try:
        # Define the 7 rank-1 tensors for 2x2x2 Strassen decomposition
        # U2, V2, W2 are 7x4 matrices
        # Indices: 00, 01, 10, 11 (row * 2 + col)
        
        U2 = np.array([
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0],# M6
            [0, 1, 0, -1] # M7
        ])
        
        V2 = np.array([
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [-1, 0, 1, 0],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ])
        
        W2 = np.array([
            [1, 0, 0, 1], # M1: C00+C11 -> W[k,i] indices: (0,0)=1, (1,1)=1
            [0, 0, 1, -1],# M2: C10-C11 -> W[1,0]=1, W[1,1]=-1
            [0, 1, 0, 1], # M3: C01+C11 -> W[0,1]=1, W[1,1]=1
            [1, 0, 1, 0], # M4: C00+C10 -> W[0,0]=1, W[1,0]=1
            [-1, 1, 0, 0],# M5: -C00+C01 -> W[0,0]=-1, W[0,1]=1
            [0, 0, 0, 1], # M6: C11 -> W[1,1]=1
            [1, 0, 0, 0]  # M7: C00 -> W[0,0]=1
        ])
        
        # Verify 2x2x2 decomposition
        # Check sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] == 1
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if not valid:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

        # Construct 4x4x4 decomposition using recursive Strassen
        # Rank 49
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        for s in range(7):
            for t in range(7):
                col_idx = s * 7 + t
                
                # U4: 4x4 matrix, flattened to 16
                # U4[I, J] = U2[s, I//2, J//2] * U2[t, I%2, J%2]
                for I in range(4):
                    for J in range(4):
                        r1 = I // 2
                        c1 = J // 2
                        r2 = I % 2
                        c2 = J % 2
                        U4[I * 4 + J, col_idx] = U2[s, r1 * 2 + c1] * U2[t, r2 * 2 + c2]
                        
                # V4: 4x4 matrix, flattened to 16
                # V4[J, K] = V2[s, J//2, K//2] * V2[t, J%2, K%2]
                for J in range(4):
                    for K in range(4):
                        r1 = J // 2
                        c1 = K // 2
                        r2 = J % 2
                        c2 = K % 2
                        V4[J * 4 + K, col_idx] = V2[s, r1 * 2 + c1] * V2[t, r2 * 2 + c2]
                        
                # W4: 4x4 matrix, flattened to 16
                # W4[K, I] = W2[s, K//2, I//2] * W2[t, K%2, I%2]
                # Note: W index is k*n+i. For 4x4, n=4.
                # W4 is indexed by (K, I) where K is row (0..3), I is col (0..3).
                # Flattened index: K * 4 + I.
                for K in range(4):
                    for I in range(4):
                        r1 = K // 2
                        c1 = I // 2
                        r2 = K % 2
                        c2 = I % 2
                        W4[K * 4 + I, col_idx] = W2[s, r1 * 2 + c1] * W2[t, r2 * 2 + c2]
        
        # Verify 4x4x4 decomposition quickly
        valid4 = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0
                    for r in range(49):
                        s += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                    if abs(s - 1) > 1e-9:
                        valid4 = False
                        break
                if not valid4: break
            if not valid4: break
            
        if not valid4:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

        # Construct 4x4x8 decomposition
        # Rank 98
        U8 = np.zeros((16, 98))
        V8 = np.zeros((32, 98))
        W8 = np.zeros((32, 98))
        
        # Part 1: k in 0..3
        U8[:, 0:49] = U4
        for r in range(49):
            for j in range(4):
                for k in range(4):
                    V8[j*8 + k, r] = V4[j*4 + k, r]
        W8[:, 0:49] = W4
        
        # Part 2: k in 4..7
        U8[:, 49:98] = U4
        for r in range(49):
            for j in range(4):
                for k in range(4):
                    V8[j*8 + 4 + k, 49 + r] = V4[j*4 + k, r]
        for r in range(49):
            for k in range(4):
                for i in range(4):
                    W8[(4+k)*4 + i, 49 + r] = W4[k*4 + i, r]
                    
        if time.time() > deadline:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
            
        return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
        
    except Exception:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
