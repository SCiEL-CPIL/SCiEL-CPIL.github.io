import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 8
    
    # Fallback: Rank 98 solution (2x Rank 49 Strassen)
    def construct_rank98():
        U2 = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1,  0, 1],
            [0, 1, 0, 0, 0,  1, 0],
            [1, 1, 0, 1, 0,  0,-1]
        ], dtype=float)
        V2 = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1,  0, 0, 1, 0],
            [0, 0, 0,  1, 0, 0, 1],
            [1, 0,-1,  0, 1, 0, 1]
        ], dtype=float)
        W2 = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1,  0, 0, 0],
            [0, 0, 1, 0,  1, 0, 0],
            [1,-1, 1, 0,  0, 1, 0]
        ], dtype=float)
        
        U4 = np.zeros((16, 49), dtype=float)
        V4 = np.zeros((16, 49), dtype=float)
        W4 = np.zeros((16, 49), dtype=float)
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        i = 2*i1 + i2
                        j = 2*j1 + j2
                        for t in range(7):
                            for s in range(7):
                                c = t*7 + s
                                U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
        
        for j1 in range(2):
            for j2 in range(2):
                for k1 in range(2):
                    for k2 in range(2):
                        j = 2*j1 + j2
                        k = 2*k1 + k2
                        for t in range(7):
                            for s in range(7):
                                c = t*7 + s
                                V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
        
        for k1 in range(2):
            for k2 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        k = 2*k1 + k2
                        i = 2*i1 + i2
                        for t in range(7):
                            for s in range(7):
                                c = t*7 + s
                                W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
        
        rank = 98
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((32, rank), dtype=float)
        W = np.zeros((32, rank), dtype=float)
        
        U[:, 0:49] = U4
        for j in range(4):
            for k in range(4):
                V[j*8+k, 0:49] = V4[j*4+k]
        for k in range(4):
            for i in range(4):
                W[k*4+i, 0:49] = W4[k*4+i]
        
        U[:, 49:98] = U4
        for j in range(4):
            for k in range(4):
                V[j*8+(k+4), 49:98] = V4[j*4+k]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i, 49:98] = W4[k*4+i]
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt Rank 38 (Laderman)
    # Laderman 4x4x4 rank 19 matrices
    # U, V, W are 16x19
    try:
        # Hardcoded Laderman matrices (example structure, need exact values)
        # If I don't have exact values, I skip this block.
        # Since I cannot guarantee exact Laderman values from memory, 
        # I will return the fallback.
        pass
    except:
        pass

    # Return fallback
    return construct_rank98()

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# To ensure a distinct and potentially better solution, 
# I will try to implement a search for lower rank? 
# No, time limit.
# I will stick to the robust Rank 98 solution.
