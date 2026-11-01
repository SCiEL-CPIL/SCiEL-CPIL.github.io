import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # Rank 42 using Strassen for 6 blocks of 2x2
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U_idx = 2*i + j
    
    # For each block k in 0..5
    for k in range(6):
        # Block offset in p dimension
        k0 = 2 * k
        k1 = 2 * k + 1
        
        # Term 1: (A00+A11)(B00+B11)
        r = k * 7 + 0
        U[0, r] += 1 # A00
        U[3, r] += 1 # A11
        V[0 * p + k0, r] += 1 # B00
        V[1 * p + k1, r] += 1 # B11
        W[k0 * n + 0, r] += 1 # C00
        W[k1 * n + 0, r] += 1 # C01
        W[k0 * n + 1, r] += 1 # C10
        W[k1 * n + 1, r] += -1 # C11
        
        # Term 2: (A10+A11)B00
        r = k * 7 + 1
        U[2, r] += 1 # A10
        U[3, r] += 1 # A11
        V[0 * p + k0, r] += 1 # B00
        W[k0 * n + 1, r] += 1 # C10
        W[k1 * n + 1, r] += -1 # C11
        
        # Term 3: A00(B01-B11)
        r = k * 7 + 2
        U[0, r] += 1 # A00
        V[0 * p + k1, r] += 1 # B01
        V[1 * p + k1, r] += -1 # B11
        W[k1 * n + 0, r] += 1 # C01
        W[k1 * n + 1, r] += 1 # C11
        
        # Term 4: A11(B10-B00)
        r = k * 7 + 3
        U[3, r] += 1 # A11
        V[1 * p + k0, r] += 1 # B10
        V[0 * p + k0, r] += -1 # B00
        W[k0 * n + 0, r] += 1 # C00
        W[k0 * n + 1, r] += 1 # C10
        
        # Term 5: (A00+A01)B11
        r = k * 7 + 4
        U[0, r] += 1 # A00
        U[1, r] += 1 # A01
        V[1 * p + k1, r] += 1 # B11
        W[k0 * n + 0, r] += -1 # C00
        W[k1 * n + 0, r] += 1 # C01
        
        # Term 6: (A10-A00)(B00+B01)
        r = k * 7 + 5
        U[2, r] += 1 # A10
        U[0, r] += -1 # A00
        V[0 * p + k0, r] += 1 # B00
        V[0 * p + k1, r] += 1 # B01
        W[k1 * n + 1, r] += 1 # C11
        
        # Term 7: (A01-A11)(B10+B11)
        r = k * 7 + 6
        U[1, r] += 1 # A01
        U[3, r] += -1 # A11
        V[1 * p + k0, r] += 1 # B10
        V[1 * p + k1, r] += 1 # B11
        W[k0 * n + 0, r] += 1 # C00
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
