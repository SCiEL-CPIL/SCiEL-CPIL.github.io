import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7) where W uses column-major indexing W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2: (A10+A11)*B00 -> C10 += M2, C11 -= M2
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[3, 1] += -1
    
    # M3: A00*(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] += -1
    W[2, 2] += 1; W[3, 2] += 1
    
    # M4: A11*(B10-B00) -> C00 += M4, C10 += M4
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] += -1
    W[0, 3] += 1; W[1, 3] += 1
    
    # M5: (A00+A01)*B11 -> C00 -= M5, C01 += M5
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += -1; W[2, 4] += 1
    
    # M6: (A10-A00)*(B00+B01) -> C11 += M6
    U[2, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # M7: (A01-A11)*(B10+B11) -> C00 += M7
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _schoolbook_3x3():
    """
    Return the rank-27 schoolbook decomposition for 3x3 matrix multiplication.
    U: (9, 27), V: (9, 27), W: (9, 27) where W uses column-major indexing W[k*3+i, r]
    """
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, r] = 1
                V[j * 3 + k, r] = 1
                W[k * 3 + i, r] = 1
                r += 1
    
    return U, V, W

def construct():
    n, m, p = 6, 6, 6
    
    U2, V2, W2 = _strassen_2x2()
    U3, V3, W3 = _schoolbook_3x3()
    
    rank2 = U2.shape[1]  # 7
    rank3 = U3.shape[1]  # 27
    
    total_rank = rank2 * rank3  # 189
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for r2 in range(rank2):
        for r3 in range(rank3):
            r = r2 * rank3 + r3
            
            # U: A[i2*3+i3, j2*3+j3] -> index (i2*3+i3)*6 + (j2*3+j3)
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            global_i = i2 * 3 + i3
                            global_j = j2 * 3 + j3
                            U[global_i * 6 + global_j, r] += U2[i2, r2] * U3[i3 * 3 + j3, r3]
            
            # V: B[j2*3+j3, k2*3+k3] -> index (j2*3+j3)*6 + (k2*3+k3)
            for j2 in range(2):
                for k2 in range(2):
                    for j3 in range(3):
                        for k3 in range(3):
                            global_j = j2 * 3 + j3
                            global_k = k2 * 3 + k3
                            V[global_j * 6 + global_k, r] += V2[j2, r2] * V3[j3 * 3 + k3, r3]
            
            # W: C[i2*3+i3, k2*3+k3] -> W index = k*6 + i = (k2*3+k3)*6 + (i2*3+i3)
            for i2 in range(2):
                for k2 in range(2):
                    for i3 in range(3):
                        for k3 in range(3):
                            global_i = i2 * 3 + i3
                            global_k = k2 * 3 + k3
                            W[global_k * 6 + global_i, r] += W2[k2 * 2 + i2, r2] * W3[k3 * 3 + i3, r3]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
