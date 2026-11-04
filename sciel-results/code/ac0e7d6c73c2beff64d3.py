import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110
    
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part L: j=0
    # A[i, 0] * B[0, k] -> C[k, i]
    # U index: i*3 + 0
    # V index: 0*3 + k
    # W index: k*2 + i
    r = 0
    for i in range(2):
        for k in range(3):
            U[i * 3 + 0][r] = 1
            V[0 * 3 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
    
    # Part R1: j in {1, 2}, k in {0, 1}
    # A[i, j+1] * B[j+1, k] -> C[k, i]
    # Strassen on 2x2x2
    # A' indices: i in {0,1}, j' in {0,1} -> global A[i, j'+1]
    # B' indices: j' in {0,1}, k' in {0,1} -> global B[j'+1, k']
    # C' indices: k' in {0,1}, i in {0,1} -> global C[k', i]
    
    # Strassen terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10+B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for U (A), V (B), W (C)
    # W is k*n + i.
    
    # Term 0 (M1)
    # U: A00(1), A11(1)
    # V: B00(1), B11(1)
    # W: C00(1), C01(-1), C10(1), C11(1) -> Wait, C indices in Strassen are local.
    # Local C00 -> global C[0, 0] (k=0, i=0) -> W[0*2+0]
    # Local C01 -> global C[1, 0] (k=1, i=0) -> W[1*2+0]
    # Local C10 -> global C[0, 1] (k=0, i=1) -> W[0*2+1]
    # Local C11 -> global C[1, 1] (k=1, i=1) -> W[1*2+1]
    
    # Let's define the local to global mapping for W
    # local (kl, il) -> global (k, i)
    # kl=0 -> k=0, kl=1 -> k=1
    # il=0 -> i=0, il=1 -> i=1
    
    # M1 contributes to:
    # C00: +1
    # C01: -1 (Wait, C01 = M3+M4? No. C01 = M3+M4 is for C[0,1]? No.)
    # Standard Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # So W coefficients for each M:
    # M1: C00:+1, C11:+1
    # M2: C10:+1, C11:-1
    # M3: C01:+1, C11:+1
    # M4: C00:-1, C01:+1
    # M5: C00:+1
    # M6: C11:+1
    # M7: C00:-1, C10:+1
    
    # U coefficients for each M (Local A indices 00, 01, 10, 11)
    # M1: A00:1, A11:1
    # M2: A10:1, A11:1
    # M3: A00:1
    # M4: A11:1
    # M5: A00:1, A01:1
    # M6: A10:1, A00:-1
    # M7: A01:1, A11:-1
    
    # V coefficients for each M (Local B indices 00, 01, 10, 11)
    # M1: B00:1, B11:1
    # M2: B00:1
    # M3: B01:1, B11:-1
    # M4: B10:1, B11:1
    # M5: B11:1
    # M6: B00:1, B01:1
    # M7: B10:1, B11:1
    
    # Mapping Local (i,j) to Global A[i, j+1]
    # Mapping Local (j,k) to Global B[j+1, k]
    # Mapping Local (k,i) to Global C[k, i] -> W[k*2+i]
    
    strassen_data = [
        # (U_coefs, V_coefs, W_coefs)
        # U_coefs: dict (local_i, local_j) -> val
        # V_coefs: dict (local_j, local_k) -> val
        # W_coefs: dict (local_k, local_i) -> val
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (1,1):1}, {(0,0):-1, (0,1):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,0):1}) # M7
    ]
    
    for d in strassen_data:
        u_coefs, v_coefs, w_coefs = d
        for (li, lj), val in u_coefs.items():
            U[li * 3 + (lj + 1)][r] += val
        for (lj, lk), val in v_coefs.items():
            V[(lj + 1) * 3 + lk][r] += val
        for (lk, li), val in w_coefs.items():
            W[lk * 2 + li][r] += val
        r += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
    # Part R2: j in {1, 2}, k=2
    # A[i, j+1] * B[j+1, 2] -> C[2, i]
    # Standard 4 terms
    for i in range(2):
        for lj in range(2):
            U[i * 3 + (lj + 1)][r] = 1
            V[(lj + 1) * 3 + 2][r] = 1
            W[2 * 2 + i][r] = 1
            r += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
