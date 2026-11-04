"""Rewritten program for matrix multiplication 4x4 x 4x9 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return U, V, W for M_2,2,2 with rank 7 (Strassen).
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0 (M0)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # r=1 (M1)
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # r=2 (M2)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # r=3 (M3)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # r=4 (M4)
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[0,4]=-1
    
    # r=5 (M5)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=-1; W[1,5]=1
    
    # r=6 (M6)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[3,6]=1; W[0,6]=-1
    
    return U, V, W

def _schoolbook_4x4x1():
    """
    Return U, V, W for M_4,4,1 with rank 16.
    U: (16, 16)
    V: (4, 16)
    W: (4, 16)
    """
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U[i * 4 + j, r] = 1.0
            V[j, r] = 1.0
            W[i, r] = 1.0
    return U, V, W

def _strassen_4x4x4():
    """
    Construct M_4,4,4 decomposition using recursive Strassen.
    Rank 49.
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank444 = 49
    U4 = np.zeros((16, rank444))
    V4 = np.zeros((16, rank444))
    W4 = np.zeros((16, rank444))
    
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            for p in range(2):
                for q in range(2):
                    u_out = U2[p * 2 + q, k]
                    if u_out != 0:
                        for i in range(2):
                            for j in range(2):
                                idx_a = (2 * p + i) * 4 + (2 * q + j)
                                u_in = U2[i * 2 + j, l]
                                U4[idx_a, col] += u_out * u_in
                                
                    v_out = V2[p * 2 + q, k]
                    if v_out != 0:
                        for i in range(2):
                            for j in range(2):
                                idx_b = (2 * p + i) * 4 + (2 * q + j)
                                v_in = V2[i * 2 + j, l]
                                V4[idx_b, col] += v_out * v_in
                                
                    w_out = W2[p * 2 + q, k]
                    if w_out != 0:
                        for i in range(2):
                            for j in range(2):
                                idx_c = (2 * q + j) * 4 + (2 * p + i)
                                w_in = W2[j * 2 + i, l]
                                W4[idx_c, col] += w_out * w_in
                                
    return U4, V4, W4

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U4, V4, W4 = _strassen_4x4x4()
    U1, V1, W1 = _schoolbook_4x4x1()
    
    r444 = U4.shape[1]
    r441 = U1.shape[1]
    total_rank = 2 * r444 + r441
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block 1
    U[:, 0:r444] = U4
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:r444] = V4[j * 4 + k, 0:r444]
    W[0:16, 0:r444] = W4[0:16, 0:r444]
    
    # Block 2
    U[:, r444:2*r444] = U4
    for j in range(4):
        for k in range(4):
            V[j * 9 + 4 + k, r444:2*r444] = V4[j * 4 + k, 0:r444]
    W[16:32, r444:2*r444] = W4[0:16, 0:r444]
    
    # Block 3
    U[:, 2*r444:2*r444+r441] = U1
    for j in range(4):
        V[j * 9 + 8, 2*r444:2*r444+r441] = V1[j, 0:r441]
    W[32:36, 2*r444:2*r444+r441] = W1[0:4, 0:r441]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
