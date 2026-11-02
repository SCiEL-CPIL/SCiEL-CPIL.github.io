import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # r=0
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    # r=2
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    # r=3
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    # r=4
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    # r=5
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    # r=6
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # Build 4x4x4 factors (Rank 49)
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    # U4[i, j, R] where R = 7*s + r
    # i = 2*i1 + i0, j = 2*j1 + j0
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
    
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]
    
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2*k1 + k0
                    i = 2*i1 + i0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # Build 4x4x8 factors
    # U shape (16, 98), V shape (32, 98), W shape (32, 98)
    # The 4x4x8 problem is two independent 4x4x4 problems (left and right halves of B)
    # U is shared (same A)
    # V and W are duplicated for the two halves
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Fill U (16x98)
    # U[i*4+j, R]
    # For R in 0..48, use U4 for first half
    # For R in 49..97, use U4 for second half (identical)
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i*4 + j, R] = U4[i, j, R]
                U[i*4 + j, 49 + R] = U4[i, j, R]
                
    # Fill V (32x98)
    # V[j*8+k, R]
    # First half (k < 4): V[j*8+k, R] = V4[j, k, R]
    # Second half (k >= 4): V[j*8+k, R] = V4[j, k-4, R]
    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j*8 + k, R] = V4[j, k, R]
                V[j*8 + (4+k), 49 + R] = V4[j, k, R]
                
    # Fill W (32x98)
    # W[k*4+i, R]
    # First half (k < 4): W[k*4+i, R] = W4[k, i, R]
    # Second half (k >= 4): W[k*4+i, R] = W4[k-4, i, R]
    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k*4 + i, R] = W4[k, i, R]
                W[(4+k)*4 + i, 49 + R] = W4[k, i, R]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
