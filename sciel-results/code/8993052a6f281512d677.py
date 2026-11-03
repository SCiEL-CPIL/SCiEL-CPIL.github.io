import numpy as np
import time

def _strassen_2x2():
    # Returns U, V, W for 2x2 * 2x2 = 2x2
    # Shapes: (4, 7), (4, 7), (4, 7)
    # U rows: a00, a01, a10, a11
    # V rows: b00, b01, b10, b11
    # W rows: c00, c01, c10, c11
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a00+a11)(b00+b11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (a10+a11)b00
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, -1]
    
    # M3 = a00(b01-b11)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, 1]
    
    # M4 = a11(b10-b00)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [-1, 0, 1, 0]
    
    # M5 = (a00+a01)b11
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 1, 0, 0]
    
    # M6 = (a10-a00)(b00+b01)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # M7 = (a01-a11)(b10+b11)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _block_strassen_3x3_3x8():
    n, m, p = 3, 3, 8
    rank = 68
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    # 1. A_2x2 * B_2x8 using Strassen (4 times)
    U_s, V_s, W_s = _strassen_2x2()
    
    for b_idx in range(4):
        for c in range(7):
            # U mapping: A[0,0]->0, A[0,1]->1, A[1,0]->3, A[1,1]->4
            U[0, col] = U_s[0, c]
            U[1, col] = U_s[1, c]
            U[3, col] = U_s[2, c]
            U[4, col] = U_s[3, c]
            
            # V mapping: B[0, 2b]->2b, B[0, 2b+1]->2b+1, B[1, 2b]->8+2b, B[1, 2b+1]->8+2b+1
            V[2*b_idx, col] = V_s[0, c]
            V[2*b_idx+1, col] = V_s[1, c]
            V[8 + 2*b_idx, col] = V_s[2, c]
            V[8 + 2*b_idx + 1, col] = V_s[3, c]
            
            # W mapping: C[0, 2b]->6b, C[0, 2b+1]->6b+3, C[1, 2b]->6b+1, C[1, 2b+1]->6b+4
            W[6*b_idx, col] = W_s[0, c]
            W[6*b_idx+3, col] = W_s[1, c]
            W[6*b_idx+1, col] = W_s[2, c]
            W[6*b_idx+4, col] = W_s[3, c]
            
            col += 1
            
    # 2. A_2x1 * B_1x8
    for i in range(2):
        for k in range(8):
            U[i*3 + 2, col] = 1
            V[16 + k, col] = 1
            W[k*3 + i, col] = 1
            col += 1

    # 3. A_1x2 * B_2x8
    for j in range(2):
        for k in range(8):
            U[6 + j, col] = 1
            V[j*8 + k, col] = 1
            W[k*3 + 2, col] = 1
            col += 1

    # 4. A_1x1 * B_1x8
    for k in range(8):
        U[8, col] = 1
        V[16 + k, col] = 1
        W[k*3 + 2, col] = 1
        col += 1

    return U, V, W

def construct():
    start_time = time.time()
    # Deadline guard: 120s limit, use 110s safety margin
    deadline = start_time + 110
    
    U, V, W = _block_strassen_3x3_3x8()
    
    return {'U': U, 'V': V, 'W': W}
