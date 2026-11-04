import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Target tensor dimensions
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We construct a rank 88 decomposition based on:
    # M(3,4,8) <= 2 * M(3,4,4)
    # M(3,4,4) <= 2 * M(3,2,4)
    # M(3,2,4) <= 2 * M(3,2,2)
    # M(3,2,2) <= 11 (7 Strassen + 4 Schoolbook)
    # Total Rank = 2 * 2 * 2 * 11 = 88

    rank = 88
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    # Base decomposition for 3x2x2 (Rank 11)
    # n_base=3, m_base=2, p_base=2
    # U_base: 6x11, V_base: 4x11, W_base: 6x11
    
    U_base = np.zeros((6, 11))
    V_base = np.zeros((4, 11))
    W_base = np.zeros((6, 11))
    
    # Indices for 3x2x2:
    # U[i*2+j]
    # V[j*2+k]
    # W[k*3+i]
    
    # Term 0: P1 = (a00+a11)(b00+b11)
    U_base[0, 0] = 1; U_base[3, 0] = 1
    V_base[0, 0] = 1; V_base[3, 0] = 1
    W_base[0, 0] = 1; W_base[4, 0] = 1 # C00, C11
    
    # Term 1: P2 = (a10+a11)b00
    U_base[2, 1] = 1; U_base[3, 1] = 1
    V_base[0, 1] = 1
    W_base[3, 1] = 1; W_base[4, 1] = -1 # C10, C11
    
    # Term 2: P3 = a00(b01-b11)
    U_base[0, 2] = 1
    V_base[1, 2] = 1; V_base[3, 2] = -1
    W_base[1, 2] = 1; W_base[4, 2] = 1 # C01, C11
    
    # Term 3: P4 = a11(b10-b00)
    U_base[3, 3] = 1
    V_base[2, 3] = 1; V_base[0, 3] = -1
    W_base[0, 3] = 1; W_base[3, 3] = 1 # C00, C10
    
    # Term 4: P5 = (a00+a01)b11
    U_base[0, 4] = 1; U_base[1, 4] = 1
    V_base[3, 4] = 1
    W_base[0, 4] = -1; W_base[1, 4] = 1 # C00, C01
    
    # Term 5: P6 = (a10-a00)(b00+b01)
    U_base[2, 5] = 1; U_base[0, 5] = -1
    V_base[0, 5] = 1; V_base[1, 5] = 1
    W_base[4, 5] = 1 # C11
    
    # Term 6: P7 = (a01-a11)(b10+b11)
    U_base[1, 6] = 1; U_base[3, 6] = -1
    V_base[2, 6] = 1; V_base[3, 6] = 1
    W_base[0, 6] = 1; W_base[4, 6] = 1 # C00, C11
    
    # Term 7: Q1 = a20*b00
    U_base[4, 7] = 1
    V_base[0, 7] = 1
    W_base[2, 7] = 1 # C20
    
    # Term 8: Q2 = a21*b10
    U_base[5, 8] = 1
    V_base[2, 8] = 1
    W_base[2, 8] = 1 # C20
    
    # Term 9: Q3 = a20*b01
    U_base[4, 9] = 1
    V_base[1, 9] = 1
    W_base[5, 9] = 1 # C21
    
    # Term 10: Q4 = a21*b11
    U_base[5, 10] = 1
    V_base[3, 10] = 1
    W_base[5, 10] = 1 # C21

    # Build 3x2x4 (Rank 22)
    # U_324: 6x22, V_324: 8x22, W_324: 12x22
    U_324 = np.zeros((6, 22))
    V_324 = np.zeros((8, 22))
    W_324 = np.zeros((12, 22))
    
    # Copy 1: k=0,1
    U_324[:, 0:11] = U_base
    # V indices: j*4+k. k=0,1
    for j in range(2):
        for k in range(2):
            V_324[j*4+k, 0:11] = V_base[j*2+k, 0:11]
    # W indices: k*3+i. k=0,1
    for k in range(2):
        for i in range(3):
            W_324[k*3+i, 0:11] = W_base[k*3+i, 0:11]
            
    # Copy 2: k=2,3
    U_324[:, 11:22] = U_base
    # V indices: j*4+k. k=2,3
    for j in range(2):
        for k in range(2, 4):
            V_324[j*4+k, 11:22] = V_base[j*2+(k-2), 0:11]
    # W indices: k*3+i. k=2,3
    for k in range(2, 4):
        for i in range(3):
            W_324[k*3+i, 11:22] = W_base[(k-2)*3+i, 0:11]

    # Build 3x4x4 (Rank 44)
    # U_344: 12x44, V_344: 16x44, W_344: 12x44
    U_344 = np.zeros((12, 44))
    V_344 = np.zeros((16, 44))
    W_344 = np.zeros((12, 44))
    
    # Copy 1: j=0,1 (A cols 0,1; B rows 0,1)
    U_344[:, 0:22] = U_324
    # V indices: j*4+k. j=0,1
    for j in range(2):
        for k in range(4):
            V_344[j*4+k, 0:22] = V_324[j*2+k, 0:22]
    # W indices: k*3+i. Same as 324
    W_344[:, 0:22] = W_324
    
    # Copy 2: j=2,3 (A cols 2,3; B rows 2,3)
    U_344[:, 22:44] = U_324
    # V indices: j*4+k. j=2,3
    for j in range(2, 4):
        for k in range(4):
            V_344[j*4+k, 22:44] = V_324[(j-2)*2+k, 0:22]
    # W indices: k*3+i. Same
    W_344[:, 22:44] = W_324

    # Build 3x4x8 (Rank 88)
    # U_348: 12x88, V_348: 32x88, W_348: 24x88
    U_348 = np.zeros((12, 88))
    V_348 = np.zeros((32, 88))
    W_348 = np.zeros((24, 88))
    
    # Copy 1: k=0..3 (B cols 0-3)
    U_348[:, 0:44] = U_344
    # V indices: j*8+k. k=0..3
    for j in range(4):
        for k in range(4):
            V_348[j*8+k, 0:44] = V_344[j*4+k, 0:44]
    # W indices: k*3+i. k=0..3
    for k in range(4):
        for i in range(3):
            W_348[k*3+i, 0:44] = W_344[k*3+i, 0:44]
            
    # Copy 2: k=4..7 (B cols 4-7)
    U_348[:, 44:88] = U_344
    # V indices: j*8+k. k=4..7
    for j in range(4):
        for k in range(4, 8):
            V_348[j*8+k, 44:88] = V_344[j*4+(k-4), 0:44]
    # W indices: k*3+i. k=4..7
    for k in range(4, 8):
        for i in range(3):
            W_348[k*3+i, 44:88] = W_344[(k-4)*3+i, 0:44]

    # Convert to lists
    return {
        "U": U_348.tolist(),
        "V": V_348.tolist(),
        "W": W_348.tolist()
    }
