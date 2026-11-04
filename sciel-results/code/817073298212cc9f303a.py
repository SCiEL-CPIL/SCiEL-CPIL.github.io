import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Attempt rank-21 decomposition
    rank = 21
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Term 1: (A00+A11)(B00+B11)
    U[0, 0] += 1; U[4, 0] += 1
    V[0, 0] += 1; V[5, 0] += 1
    W[0, 0] += 1; W[4, 0] += 1
    
    # Term 2: (A10+A11)B00
    U[3, 1] += 1; U[4, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[4, 1] += 1
    
    # Term 3: A00(B01-B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[5, 2] += -1
    W[3, 2] += 1; W[4, 2] += 1
    
    # Term 4: A11(B10-B00)
    U[4, 3] += 1
    V[4, 3] += 1; V[0, 3] += -1
    W[0, 3] += 1; W[1, 3] += 1
    
    # Term 5: (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[5, 4] += 1
    W[3, 4] += 1; W[0, 4] += 1
    
    # Term 6: (A10-A00)(B00+B01)
    U[3, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[4, 5] += 1
    
    # Term 7: (A01-A11)(B10+B11)
    U[1, 6] += 1; U[4, 6] += -1
    V[4, 6] += 1; V[5, 6] += 1
    W[0, 6] += 1
    
    # Terms 8-21: Cover remaining indices
    # T8: A02*B20 -> C00
    U[2, 7] += 1; V[8, 7] += 1; W[0, 7] += 1
    # T9: A02*B21 -> C01
    U[2, 8] += 1; V[9, 8] += 1; W[3, 8] += 1
    # T10: A02*B22 -> C02
    U[2, 9] += 1; V[10, 9] += 1; W[6, 9] += 1
    # T11: A02*B23 -> C03
    U[2, 10] += 1; V[11, 10] += 1; W[9, 10] += 1
    # T12: A12*B20 -> C10
    U[5, 11] += 1; V[8, 11] += 1; W[1, 11] += 1
    # T13: A12*B21 -> C11
    U[5, 12] += 1; V[9, 12] += 1; W[4, 12] += 1
    # T14: A12*B22 -> C12
    U[5, 13] += 1; V[10, 13] += 1; W[7, 13] += 1
    # T15: A12*B23 -> C13
    U[5, 14] += 1; V[11, 14] += 1; W[10, 14] += 1
    # T16: A20*B00 -> C20
    U[6, 15] += 1; V[0, 15] += 1; W[2, 15] += 1
    # T17: A20*B01 -> C21
    U[6, 16] += 1; V[1, 16] += 1; W[5, 16] += 1
    # T18: A20*B02 -> C22
    U[6, 17] += 1; V[2, 17] += 1; W[8, 17] += 1
    # T19: A20*B03 -> C23
    U[6, 18] += 1; V[3, 18] += 1; W[11, 18] += 1
    # T20: A21*B10 -> C20
    U[7, 19] += 1; V[4, 19] += 1; W[2, 19] += 1
    # T21: A21*B11 -> C21
    U[7, 20] += 1; V[5, 20] += 1; W[5, 20] += 1
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid and (time.time() - start_time) < 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=float)
    V_fb = np.zeros((m * p, rank_fb), dtype=float)
    W_fb = np.zeros((p * n, rank_fb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
