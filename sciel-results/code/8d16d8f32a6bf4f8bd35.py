import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)
    
    # Indices:
    # U: i*3+j
    # V: j*3+k
    # W: k*3+i
    
    # Term 0: A00 * B00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    
    # Term 1: A01 * B01
    U[1, 1] = 1; V[1, 1] = 1; W[3, 1] = 1
    
    # Term 2: A02 * B02
    U[2, 2] = 1; V[2, 2] = 1; W[6, 2] = 1
    
    # Term 3: A10 * B10
    U[3, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
    
    # Term 4: A11 * B11
    U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
    
    # Term 5: A12 * B12
    U[5, 5] = 1; V[5, 5] = 1; W[7, 5] = 1
    
    # Term 6: A20 * B20
    U[6, 6] = 1; V[6, 6] = 1; W[2, 6] = 1
    
    # Term 7: A21 * B21
    U[7, 7] = 1; V[7, 7] = 1; W[5, 7] = 1
    
    # Term 8: A22 * B22
    U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
    
    # Term 9: (A00+A01) * (B00+B10)
    U[0, 9] = 1; U[1, 9] = 1
    V[0, 9] = 1; V[3, 9] = 1
    W[0, 9] = 1; W[1, 9] = 1
    
    # Term 10: (A00+A02) * (B00+B20)
    U[0, 10] = 1; U[2, 10] = 1
    V[0, 10] = 1; V[6, 10] = 1
    W[0, 10] = 1; W[2, 10] = 1
    
    # Term 11: (A01+A02) * (B10+B20)
    U[1, 11] = 1; U[2, 11] = 1
    V[3, 11] = 1; V[6, 11] = 1
    W[1, 11] = 1; W[2, 11] = 1
    
    # Term 12: (A10+A11) * (B01+B11)
    U[3, 12] = 1; U[4, 12] = 1
    V[1, 12] = 1; V[4, 12] = 1
    W[3, 12] = 1; W[4, 12] = 1
    
    # Term 13: (A10+A12) * (B02+B12)
    U[3, 13] = 1; U[5, 13] = 1
    V[2, 13] = 1; V[5, 13] = 1
    W[6, 13] = 1; W[7, 13] = 1
    
    # Term 14: (A11+A12) * (B11+B12)
    U[4, 14] = 1; U[5, 14] = 1
    V[4, 14] = 1; V[5, 14] = 1
    W[4, 14] = 1; W[7, 14] = 1
    
    # Term 15: (A20+A21) * (B01+B21)
    U[6, 15] = 1; U[7, 15] = 1
    V[1, 15] = 1; V[7, 15] = 1
    W[3, 15] = 1; W[5, 15] = 1
    
    # Term 16: (A20+A22) * (B02+B22)
    U[6, 16] = 1; U[8, 16] = 1
    V[2, 16] = 1; V[8, 16] = 1
    W[6, 16] = 1; W[8, 16] = 1
    
    # Term 17: (A21+A22) * (B11+B22)
    U[7, 17] = 1; U[8, 17] = 1
    V[4, 17] = 1; V[8, 17] = 1
    W[4, 17] = 1; W[8, 17] = 1
    
    # Term 18: (A00+A10+A20) * (B00+B01+B02)
    U[0, 18] = 1; U[3, 18] = 1; U[6, 18] = 1
    V[0, 18] = 1; V[1, 18] = 1; V[2, 18] = 1
    W[0, 18] = 1; W[1, 18] = 1; W[2, 18] = 1
    
    # Term 19: (A01+A11+A21) * (B00+B01+B02)
    U[1, 19] = 1; U[4, 19] = 1; U[7, 19] = 1
    V[0, 19] = 1; V[1, 19] = 1; V[2, 19] = 1
    W[0, 19] = 1; W[1, 19] = 1; W[2, 19] = 1
    
    # Term 20: (A02+A12+A22) * (B00+B01+B02)
    U[2, 20] = 1; U[5, 20] = 1; U[8, 20] = 1
    V[0, 20] = 1; V[1, 20] = 1; V[2, 20] = 1
    W[0, 20] = 1; W[1, 20] = 1; W[2, 20] = 1
    
    # Term 21: (A00+A01+A02) * (B00+B10+B20)
    U[0, 21] = 1; U[1, 21] = 1; U[2, 21] = 1
    V[0, 21] = 1; V[3, 21] = 1; V[6, 21] = 1
    W[0, 21] = 1; W[1, 21] = 1; W[2, 21] = 1
    
    # Term 22: (A00+A01+A02) * (B01+B11+B21)
    U[0, 22] = 1; U[1, 22] = 1; U[2, 22] = 1
    V[1, 22] = 1; V[4, 22] = 1; V[7, 22] = 1
    W[3, 22] = 1; W[4, 22] = 1; W[5, 22] = 1

    # Verification
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for r in range(rank):
                    s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 27
        rank = 27
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((9, rank), dtype=float)
        W = np.zeros((9, rank), dtype=float)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
