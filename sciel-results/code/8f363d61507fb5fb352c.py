import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 3, 3, 7
    
    # Rank = 62: Use Strassen for the 2x2x2 sub-block (indices i,j,k in {0,1}),
    # and schoolbook for all remaining 55 terms.
    rank = 62
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen decomposition for the 2x2x2 sub-block (i,j,k in {0,1})
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*7 + k
    # W index: k*n + i = k*3 + i
    
    # P1: A00+A11, B00+B11, C00+=1, C11+=1
    U[0, 0] += 1  # A00 (i=0,j=0)
    U[4, 0] += 1  # A11 (i=1,j=1)
    V[0, 0] += 1  # B00 (j=0,k=0)
    V[8, 0] += 1  # B11 (j=1,k=1)
    W[0, 0] += 1  # C00 (k=0,i=0)
    W[4, 0] += 1  # C11 (k=1,i=1)
    
    # P2: A10+A11, B00, C10+=1, C11-=1
    U[3, 1] += 1  # A10 (i=1,j=0)
    U[4, 1] += 1  # A11 (i=1,j=1)
    V[0, 1] += 1  # B00 (j=0,k=0)
    W[3, 1] += 1  # C10 (k=1,i=0)
    W[4, 1] -= 1  # C11 (k=1,i=1)
    
    # P3: A00, B01-B11, C01+=1, C11+=1
    U[0, 2] += 1  # A00 (i=0,j=0)
    V[7, 2] += 1  # B01 (j=0,k=1)
    V[8, 2] -= 1  # B11 (j=1,k=1)
    W[1, 2] += 1  # C01 (k=0,i=1)
    W[4, 2] += 1  # C11 (k=1,i=1)
    
    # P4: A11, B10-B00, C00+=1, C10+=1
    U[4, 3] += 1  # A11 (i=1,j=1)
    V[14, 3] += 1  # B10 (j=1,k=0)
    V[0, 3] -= 1  # B00 (j=0,k=0)
    W[0, 3] += 1  # C00 (k=0,i=0)
    W[3, 3] += 1  # C10 (k=1,i=0)
    
    # P5: A00+A01, B11, C00-=1, C01+=1
    U[0, 4] += 1  # A00 (i=0,j=0)
    U[1, 4] += 1  # A01 (i=0,j=1)
    V[8, 4] += 1  # B11 (j=1,k=1)
    W[0, 4] -= 1  # C00 (k=0,i=0)
    W[1, 4] += 1  # C01 (k=0,i=1)
    
    # P6: A10-A00, B00+B01, C11+=1
    U[3, 5] += 1  # A10 (i=1,j=0)
    U[0, 5] -= 1  # A00 (i=0,j=0)
    V[0, 5] += 1  # B00 (j=0,k=0)
    V[7, 5] += 1  # B01 (j=0,k=1)
    W[4, 5] += 1  # C11 (k=1,i=1)
    
    # P7: A01-A11, B10+B11, C00+=1, C11+=1
    U[1, 6] += 1  # A01 (i=0,j=1)
    U[4, 6] -= 1  # A11 (i=1,j=1)
    V[14, 6] += 1  # B10 (j=1,k=0)
    V[8, 6] += 1  # B11 (j=1,k=1)
    W[0, 6] += 1  # C00 (k=0,i=0)
    W[4, 6] += 1  # C11 (k=1,i=1)
    
    # Schoolbook terms for remaining (i,j,k) not in {0,1}^3
    r_idx = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                U[i * m + j, r_idx] = 1
                V[j * p + k, r_idx] = 1
                W[k * n + i, r_idx] = 1
                r_idx += 1
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
