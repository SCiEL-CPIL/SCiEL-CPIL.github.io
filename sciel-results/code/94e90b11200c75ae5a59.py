import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    
    # Strassen 2x2x2 rank-7 decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (c+d)*e -> C[1,0]+=1, C[1,1]-=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = a*(f-h) -> C[0,1]+=1, C[1,1]+=1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = d*(g-e) -> C[0,0]+=1, C[1,0]+=1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (a+b)*h -> C[0,0]-=1, C[0,1]+=1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (c-a)*(e+f) -> C[1,1]+=1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (b-d)*(g+h) -> C[0,0]+=1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Build 4x4x4 rank-49 decomposition using Kronecker product
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Indices: i, j, k in 0..3
    # i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # r = 7*r1 + r2
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    i = i1 * 2 + i0
                    j = j1 * 2 + j0
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            U4[i * 4 + j, r] = U2[i1 * 2 + j1, r1] * U2[i0 * 2 + j0, r2]
                            
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    j = j1 * 2 + j0
                    k = k1 * 2 + k0
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            V4[j * 4 + k, r] = V2[j1 * 2 + k1, r1] * V2[j0 * 2 + k0, r2]
                            
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    k = k1 * 2 + k0
                    i = i1 * 2 + i0
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            W4[k * 4 + i, r] = W2[k1 * 2 + i1, r1] * W2[k0 * 2 + i0, r2]
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # U is same for both halves of B (left and right 4x4 blocks)
    U[:, :49] = U4
    U[:, 49:] = U4
    
    # V: 4x8 matrix B.
    # V rows: j*8+k. j in 0..3, k in 0..7.
    # Left half (k=0..3) uses V4. Right half (k=4..7) uses V4.
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, :49] = V4[j * 4 + k]
            V[j * 8 + (k + 4), 49:] = V4[j * 4 + k]
            
    # W: 8x4 matrix C.
    # W rows: k*4+i. k in 0..7, i in 0..3.
    # Left half (k=0..3) uses W4. Right half (k=4..7) uses W4.
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W4[k * 4 + i]
            W[(k + 4) * 4 + i, 49:] = W4[k * 4 + i]
            
    if time.time() - start_time > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
