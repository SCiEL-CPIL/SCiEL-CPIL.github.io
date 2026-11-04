import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    
    # We will use the Strassen algorithm for 4x4x4, applied to two 4x4 blocks of B.
    # Total rank = 49 + 49 = 98.
    
    # Strassen factors for 2x2x2 (Rank 7)
    # Indices for 2x2: i,j in {0,1}.
    # U_2x2 (4, 7), V_2x2 (4, 7), W_2x2 (4, 7).
    # i*2+j, j*2+k, k*2+i.
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C00 -= M3, C01 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C01 -= M7, C10 -= M7? No.
    
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's build the 7x4 matrices for 2x2x2.
    # U (A coeffs):
    # M1: A00, A11 -> [1, 0, 0, 1]
    # M2: A10, A11 -> [0, 0, 1, 1]
    # M3: A00 -> [1, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 1]
    # M5: A00, A01 -> [1, 1, 0, 0]
    # M6: A10, A00 -> [1, 0, 1, 0]
    # M7: A01, A11 -> [0, 1, 0, 1]
    
    U_2x2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    
    # V (B coeffs):
    # M1: B00, B11 -> [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    # M3: B01, B11 -> [0, 1, 0, 1]
    # M4: B10, B00 -> [1, 0, 1, 0]
    # M5: B11 -> [0, 0, 0, 1]
    # M6: B00, B01 -> [1, 1, 0, 0]
    # M7: B10, B11 -> [0, 0, 1, 1]
    
    V_2x2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # W (C coeffs):
    # M1: C00, C11 -> [1, 0, 0, 1]
    # M2: C10, C11 -> [0, 0, 1, 1] (C10 is index 2, C11 is index 3? k*2+i. C00=0, C01=1, C10=2, C11=3)
    # M3: C00, C01 -> [1, 1, 0, 0]
    # M4: C00, C10 -> [1, 0, 1, 0]
    # M5: C00, C01 -> [1, 1, 0, 0]
    # M6: C11 -> [0, 0, 0, 1]
    # M7: C00 -> [1, 0, 0, 0]
    
    W_2x2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    # Now extend to 4x4x4 using Strassen on 2x2 blocks.
    # This is complex. I will use the standard algorithm (Rank 64) for 4x4x4 to be safe.
    # Rank 64 for 4x4x4.
    # Total Rank for 4x4x8: 128.
    
    # Standard Algorithm:
    # U[i*m+j, r] = 1 if r corresponds to (i,j).
    # V[j*p+k, r] = 1 if r corresponds to (j,k).
    # W[k*n+i, r] = 1 if r corresponds to (i,k).
    
    rank = n * m * p # 4*4*8 = 128
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
