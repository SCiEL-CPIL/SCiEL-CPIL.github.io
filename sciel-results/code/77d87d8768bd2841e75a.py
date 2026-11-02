import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen factors for 2x2x2
    # U2[i, j, r], V2[j, k, r], W2[k, i, r]
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # 2. Build 4x4x4 factors using tensor product (Kronecker)
    # U4[i, j, R], V4[j, k, R], W4[k, i, R]
    # R = 7*s + r
    # We can compute this via reshaping and multiplying
    
    # U4 shape (4, 4, 49)
    # U2 shape (2, 2, 7)
    # U4[i, j, 7s+r] = U2[i//2, j//2, s] * U2[i%2, j%2, r]
    
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
                    
    for j in range(4):
        for k in range(4):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]
                    
    for k in range(4):
        for i in range(4):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # 3. Build 4x4x8 factors
    # Rank 98
    # U shape (16, 98)
    # V shape (32, 98)
    # W shape (32, 98)
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # U is duplicated
    U[:, :49] = U4.reshape(16, 49)
    U[:, 49:] = U4.reshape(16, 49)
    
    # V: First 16 rows (k<4) use first 49 cols. Last 16 rows (k>=4) use last 49 cols.
    # V index: j*8 + k
    # For k in 0..3: V[j*8+k, r] = V4[j, k, r]
    # For k in 4..7: V[j*8+k, 49+r] = V4[j, k-4, r]
    
    V4_flat = V4.reshape(16, 49)
    
    # Fill V
    for j in range(4):
        for k in range(4):
            idx = j * 8 + k
            V[idx, :49] = V4[j, k, :]
            
            idx2 = j * 8 + (4 + k)
            V[idx2, 49:] = V4[j, k, :]
            
    # W: First 16 rows (k<4) use first 49 cols. Last 16 rows (k>=4) use last 49 cols.
    # W index: k*4 + i
    # For k in 0..3: W[k*4+i, r] = W4[k, i, r]
    # For k in 4..7: W[k*4+i, 49+r] = W4[k-4, i, r]
    
    for k in range(4):
        for i in range(4):
            idx = k * 4 + i
            W[idx, :49] = W4[k, i, :]
            
            idx2 = (4 + k) * 4 + i
            W[idx2, 49:] = W4[k, i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
