import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strassen 2x2x2
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ], dtype=int)
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 1],
        [1, 0, -1, 0, 1, 0, 0]
    ], dtype=int)
    
    S_W = np.array([
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0]
    ], dtype=int)
    
    # Schoolbook 2x2x3
    # A is 2x2, B is 2x3, C is 2x3
    # Terms:
    # 1: A[0,0] * B[0,:] -> C[0,:]
    # 2: A[0,1] * B[1,:] -> C[0,:]
    # 3: A[1,0] * B[0,:] -> C[1,:]
    # 4: A[1,1] * B[1,:] -> C[1,:]
    
    K_U = np.eye(4, dtype=int)
    
    # B indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # Term 1: B[0,0], B[0,1], B[0,2] are 1. Others 0.
    # Term 2: B[1,0], B[1,1], B[1,2] are 1. Others 0.
    # Term 3: B[0,0], B[0,1], B[0,2] are 1. Others 0.
    # Term 4: B[1,0], B[1,1], B[1,2] are 1. Others 0.
    
    K_V = np.zeros((6, 4), dtype=int)
    K_V[0:3, 0] = 1
    K_V[3:6, 1] = 1
    K_V[0:3, 2] = 1
    K_V[3:6, 3] = 1
    
    # C indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # Term 1: C[0,0], C[0,1], C[0,2] are 1.
    # Term 2: C[0,0], C[0,1], C[0,2] are 1.
    # Term 3: C[1,0], C[1,1], C[1,2] are 1.
    # Term 4: C[1,0], C[1,1], C[1,2] are 1.
    
    K_W = np.zeros((6, 4), dtype=int)
    K_W[0:3, 0] = 1
    K_W[0:3, 1] = 1
    K_W[3:6, 2] = 1
    K_W[3:6, 3] = 1
    
    # Combine via Kronecker Product
    # U shape: (4*4, 7*4) = (16, 28)
    # V shape: (4*6, 7*4) = (24, 28)
    # W shape: (4*6, 7*4) = (24, 28)
    
    U = np.zeros((16, 28), dtype=int)
    V = np.zeros((24, 28), dtype=int)
    W = np.zeros((24, 28), dtype=int)
    
    for r in range(7):
        for s in range(4):
            col = r * 4 + s
            
            # U
            # S_U[i1, j1, r] * K_U[i2, j2, s]
            # Row index: (i1*2 + i2)*4 + (j1*2 + j2)
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            val = S_U[i1, j1, r] * K_U[i2, j2, s]
                            if val != 0:
                                row = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                                U[row, col] += val
            
            # V
            # S_V[j1, k1, r] * K_V[j2, k2, s]
            # B index: j = j1*2 + j2, k = k1*3 + k2
            # Row index: j*6 + k
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(3):
                            val = S_V[j1, k1, r] * K_V[j2, k2, s]
                            if val != 0:
                                j = j1 * 2 + j2
                                k = k1 * 3 + k2
                                row = j * 6 + k
                                V[row, col] += val
            
            # W
            # S_W[i1, k1, r] * K_W[i2, k2, s]
            # C index: i = i1*2 + i2, k = k1*3 + k2
            # Row index: i*6 + k
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(3):
                            val = S_W[i1, k1, r] * K_W[i2, k2, s]
                            if val != 0:
                                i = i1 * 2 + i2
                                k = k1 * 3 + k2
                                row = i * 6 + k
                                W[row, col] += val
                                
    return {'U': U, 'V': V, 'W': W}
