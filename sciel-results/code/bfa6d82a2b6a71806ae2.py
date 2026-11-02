import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=6
    # Rank 21 decomposition using 3 Strassen blocks
    rank = 21
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01+B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    
    # U coefficients for A (4 entries per term)
    # V coefficients for B (4 entries per term)
    # W coefficients for C (4 entries per term)
    
    strassen_U = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, -1, 1, 0],# M3: B10-B11
        [0, 1, 0, 1], # M4: B01+B11
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B10
        [0, 1, 0, 1]  # M7: B11+B01
    ]
    
    strassen_W = [
        [1, 0, 0, -1], # C00 = M1+M4-M5-M7
        [0, 1, 0, 0],  # C01 = M3+M5
        [0, 0, 1, 0],  # C10 = M2+M4
        [1, -1, 1, 1]  # C11 = M1-M2+M3+M7
    ]
    # Wait, W in Strassen is usually defined as C = sum W_r M_r.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # So W[0] (for C00) should be [1, 0, 0, 1, -1, 0, -1] corresponding to M1..M7?
    # Let's re-index W to match the 7 terms.
    
    # Term indices: 0:M1, 1:M2, 2:M3, 3:M4, 4:M5, 5:M6, 6:M7
    # C00: M1, M4, M5, M7 -> coeffs: 1, 0, 0, 1, -1, 0, -1
    # C01: M3, M5 -> coeffs: 0, 0, 1, 0, 1, 0, 0
    # C10: M2, M4 -> coeffs: 0, 1, 0, 1, 0, 0, 0
    # C11: M1, M2, M3, M7 -> coeffs: 1, -1, 1, 0, 0, 0, 1
    
    W_coeffs = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 0, 1]   # C11
    ]
    
    for block in range(3):
        # Block 0: k=0,1
        # Block 1: k=2,3
        # Block 2: k=4,5
        
        for r in range(7):
            col = block * 7 + r
            
            # Fill U
            # U is (n*m, rank) = (4, 21)
            # Indices 0..3 correspond to A00, A01, A10, A11
            U[0, col] = strassen_U[r][0]
            U[1, col] = strassen_U[r][1]
            U[2, col] = strassen_U[r][2]
            U[3, col] = strassen_U[r][3]
            
            # Fill V
            # V is (m*p, rank) = (12, 21)
            # Indices: j*p + k
            # j=0,1; k=0..5
            # For block, k_local is 0,1. k_global is block*2 + k_local
            # V index = j*6 + k_global
            # strassen_V[r] has 4 entries: B00, B01, B10, B11 (local)
            # Local B00 -> j=0, k=0
            # Local B01 -> j=0, k=1
            # Local B10 -> j=1, k=0
            # Local B11 -> j=1, k=1
            
            k_base = block * 2
            # B00
            V[0 * 6 + (k_base + 0), col] = strassen_V[r][0]
            # B01
            V[0 * 6 + (k_base + 1), col] = strassen_V[r][1]
            # B10
            V[1 * 6 + (k_base + 0), col] = strassen_V[r][2]
            # B11
            V[1 * 6 + (k_base + 1), col] = strassen_V[r][3]
            
            # Fill W
            # W is (p*n, rank) = (12, 21)
            # Indices: k*n + i
            # i=0,1; k=0..5
            # W index = k*2 + i
            # strassen_W corresponds to C00, C01, C10, C11
            # C00 -> i=0, k=0
            # C01 -> i=0, k=1
            # C10 -> i=1, k=0
            # C11 -> i=1, k=1
            
            # W_coeffs[0] is for C00
            # W_coeffs[1] is for C01
            # W_coeffs[2] is for C10
            # W_coeffs[3] is for C11
            
            # C00: i=0, k=k_base
            W[(k_base + 0) * 2 + 0, col] = W_coeffs[0][r]
            # C01: i=0, k=k_base+1
            W[(k_base + 1) * 2 + 0, col] = W_coeffs[1][r]
            # C10: i=1, k=k_base
            W[(k_base + 0) * 2 + 1, col] = W_coeffs[2][r]
            # C11: i=1, k=k_base+1
            W[(k_base + 1) * 2 + 1, col] = W_coeffs[3][r]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
