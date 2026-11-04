import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    # 3x3 Schoolbook Factors
    U3 = np.zeros((9, 9), dtype=int)
    V3 = np.zeros((9, 9), dtype=int)
    W3 = np.zeros((9, 9), dtype=int)
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                a_idx = i * 3 + k
                b_idx = k * 3 + j
                c_idx = i * 3 + j
                U3[a_idx, idx] = 1
                V3[b_idx, idx] = 1
                W3[c_idx, idx] = 1
                idx += 1
                
    # 2x2 Strassen Factors
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Kronecker Products
    U6_kron = np.kron(U3, U2)
    V6_kron = np.kron(V3, V2)
    W6_kron = np.kron(W3, W2)
    
    # Permutation
    perm = np.zeros(36, dtype=int)
    for I in range(3):
        for K in range(3):
            for i in range(2):
                for k in range(2):
                    idx_kron = (I * 3 + K) * 4 + (i * 2 + k)
                    idx_rm = (3 * I + i) * 6 + (3 * K + k)
                    perm[idx_kron] = idx_rm
                    
    U6 = U6_kron[perm, :]
    V6 = V6_kron[perm, :]
    W6 = W6_kron[perm, :]
    
    return {'U': U6, 'V': V6, 'W': W6}
