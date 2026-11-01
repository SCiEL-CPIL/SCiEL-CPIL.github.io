import numpy as np

def construct():
    n, m, p = 2, 2, 16
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for A (i,j) -> index i*m+j
    # M1: A00 + A11 -> [1, 0, 0, 1]
    # M2: A10 + A11 -> [0, 0, 1, 1]
    # M3: A00      -> [1, 0, 0, 0]
    # M4: A11      -> [0, 0, 0, 1]
    # M5: A00 + A01-> [1, 1, 0, 0]
    # M6: A10 - A00-> [0, 0, 1, -1]
    # M7: A11 - A01-> [0, 0, -1, 1]
    
    A_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 0, -1, 1]
    ]
    
    # B coefficients for each M, relative to block (j, k_local)
    # j in {0,1}, k_local in {0,1}
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    B_coeffs = [
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    # W coefficients for each M, relative to block (i, k_local)
    # M1: C00 + C11
    # M2: C10 - C11
    # M3: C01 + C11
    # M4: C00 + C10
    # M5: -C00 + C01
    # M6: C11
    # M7: C00
    
    W_coeffs = [
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    num_blocks = p // 2
    for m_block in range(num_blocks):
        for s in range(7):
            r = m_block * 7 + s
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = A_coeffs[s][i * m + j]
            
            # Fill V
            for (j, k_loc), val in B_coeffs[s].items():
                k = m_block * 2 + k_loc
                V[j * p + k, r] = val
            
            # Fill W
            for (i, k_loc), val in W_coeffs[s].items():
                k = m_block * 2 + k_loc
                W[k * n + i, r] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
