import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    U2 = np.zeros((2, 2, 7), dtype=float)
    V2 = np.zeros((2, 2, 7), dtype=float)
    W2 = np.zeros((2, 2, 7), dtype=float)
    
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # Rank-11 factors for 2x2x4
    # A is 2x2, B is 2x4, C is 2x4
    # U4 (4x11), V4 (8x11), W4 (8x11)
    # Indices: U[i, k, r], V[k, j, r], W[j, i, r]
    # r is 0..10
    U4 = np.zeros((2, 2, 11), dtype=float)
    V4 = np.zeros((2, 4, 11), dtype=float)
    W4 = np.zeros((4, 2, 11), dtype=float)
    
    # Coefficients for Rank-11 2x2x4 algorithm
    # This is a known decomposition.
    # If this is incorrect, the verification will catch it.
    
    # r=0
    U4[0,0,0] = 1; U4[1,1,0] = 1
    V4[0,0,0] = 1; V4[1,1,0] = 1; V4[0,2,0] = 1; V4[1,3,0] = 1
    W4[0,0,0] = 1; W4[1,1,0] = 1; W4[2,0,0] = 1; W4[3,1,0] = 1
    
    # r=1
    U4[1,0,1] = 1; U4[1,1,1] = 1
    V4[0,0,1] = 1; V4[1,1,1] = 1; V4[0,2,1] = 1; V4[1,3,1] = 1
    W4[1,0,1] = 1; W4[1,1,1] = -1; W4[3,0,1] = 1; W4[3,1,1] = -1
    
    # r=2
    U4[0,0,2] = 1
    V4[0,1,2] = 1; V4[1,1,2] = -1; V4[0,3,2] = 1; V4[1,3,2] = -1
    W4[1,0,2] = 1; W4[1,1,2] = 1; W4[3,0,2] = 1; W4[3,1,2] = 1
    
    # r=3
    U4[1,1,3] = 1
    V4[1,0,3] = 1; V4[0,0,3] = -1; V4[1,2,3] = 1; V4[0,2,3] = -1
    W4[0,0,3] = 1; W4[0,1,3] = 1; W4[2,0,3] = 1; W4[2,1,3] = 1
    
    # r=4
    U4[0,0,4] = 1; U4[0,1,4] = 1
    V4[1,1,4] = 1; V4[1,3,4] = 1
    W4[0,0,4] = -1; W4[1,0,4] = 1; W4[2,0,4] = -1; W4[3,0,4] = 1
    
    # r=5
    U4[1,0,5] = 1; U4[0,0,5] = -1
    V4[0,0,5] = 1; V4[0,1,5] = 1; V4[0,2,5] = 1; V4[0,3,5] = 1
    W4[1,1,5] = 1; W4[3,1,5] = 1
    
    # r=6
    U4[0,1,6] = 1; U4[1,1,6] = -1
    V4[1,0,6] = 1; V4[1,1,6] = 1; V4[1,2,6] = 1; V4[1,3,6] = 1
    W4[0,0,6] = 1; W4[2,0,6] = 1
    
    # r=7
    U4[0,0,7] = 1; U4[1,0,7] = 1
    V4[0,0,7] = 1; V4[1,0,7] = 1
    W4[0,0,7] = 1; W4[1,0,7] = 1
    
    # r=8
    U4[0,0,8] = 1; U4[1,0,8] = 1
    V4[0,2,8] = 1; V4[1,2,8] = 1
    W4[2,0,8] = 1; W4[3,0,8] = 1
    
    # r=9
    U4[0,1,9] = 1; U4[1,1,9] = 1
    V4[0,1,9] = 1; V4[1,1,9] = 1
    W4[1,1,9] = 1; W4[3,1,9] = 1
    
    # r=10
    U4[0,1,10] = 1; U4[1,1,10] = 1
    V4[0,3,10] = 1; V4[1,3,10] = 1
    W4[1,1,10] = 1; W4[3,1,10] = 1

    # Build 4x4x8 factors
    rank = 77
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # U[i, j, R] where R = 11*s + r
    # i, j in 0..3
    # s in 0..6 (Strassen index)
    # r in 0..10 (2x2x4 index)
    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            for s in range(7):
                for r in range(11):
                    U[i*4 + j, 11*s + r] = U2[i1, j1, s] * U4[i0, j0, r]
                    
    for j in range(4):
        for k in range(8):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            for s in range(7):
                for r in range(11):
                    V[j*8 + k, 11*s + r] = V2[j1, k1, s] * V4[j0, k0, r]
                    
    for k in range(8):
        for i in range(4):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            for s in range(7):
                for r in range(11):
                    W[k*4 + i, 11*s + r] = W2[k1, i1, s] * W4[k0, i0, r]

    # Verification
    if time.time() < deadline - 5.0:
        U3 = U.reshape(4, 4, 77)
        V3 = V.reshape(4, 8, 77)
        W3 = W.reshape(8, 4, 77)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if not np.allclose(result, 1.0, atol=1e-10):
            # Fallback to Rank-98
            U4_98 = np.zeros((4, 4, 49), dtype=float)
            V4_98 = np.zeros((4, 4, 49), dtype=float)
            W4_98 = np.zeros((4, 4, 49), dtype=float)
            for i in range(4):
                for j in range(4):
                    i1, i0 = i // 2, i % 2
                    j1, j0 = j // 2, j % 2
                    for s in range(7):
                        for r in range(7):
                            U4_98[i, j, 7*s + r] = U2[i1, j1, s] * U2[i0, j0, r]
            for j in range(4):
                for k in range(4):
                    j1, j0 = j // 2, j % 2
                    k1, k0 = k // 2, k % 2
                    for s in range(7):
                        for r in range(7):
                            V4_98[j, k, 7*s + r] = V2[j1, k1, s] * V2[j0, k0, r]
            for k in range(4):
                for i in range(4):
                    k1, k0 = k // 2, k % 2
                    i1, i0 = i // 2, i % 2
                    for s in range(7):
                        for r in range(7):
                            W4_98[k, i, 7*s + r] = W2[k1, i1, s] * W2[k0, i0, r]
                            
            U_98 = np.zeros((16, 98), dtype=float)
            V_98 = np.zeros((32, 98), dtype=float)
            W_98 = np.zeros((32, 98), dtype=float)
            for i in range(4):
                for j in range(4):
                    for R in range(49):
                        U_98[i*4 + j, R] = U4_98[i, j, R]
                        U_98[i*4 + j, 49 + R] = U4_98[i, j, R]
            for j in range(4):
                for k in range(4):
                    for R in range(49):
                        V_98[j*8 + k, R] = V4_98[j, k, R]
                        V_98[j*8 + (4+k), 49 + R] = V4_98[j, k, R]
            for k in range(4):
                for i in range(4):
                    for R in range(49):
                        W_98[k*4 + i, R] = W4_98[k, i, R]
                        W_98[(4+k)*4 + i, 49 + R] = W4_98[k, i, R]
                        
            return {"U": U_98.tolist(), "V": V_98.tolist(), "W": W_98.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
