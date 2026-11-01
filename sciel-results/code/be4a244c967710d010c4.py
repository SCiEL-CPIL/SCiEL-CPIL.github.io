"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 blocks * 7 Strassen terms = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # Indices: U: i*2+j, V: j*2+k', W: k'*2+i
    # t=0..6 corresponds to M1..M7
    
    # U coefficients (4 entries each: 00, 01, 10, 11)
    U_coeffs = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [-1, 0, 1, 0],# M6: A10, A00
        [0, 1, 0, -1] # M7: A01, A11
    ]
    
    # V coefficients (4 entries each: 00, 01, 10, 11)
    V_coeffs = [
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01, B11
        [-1, 0, 1, 0],# M4: B10, B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 0, 1, 1]  # M7: B10, B11
    ]
    
    # W coefficients (4 entries each: 00, 01, 10, 11)
    # W index is k'*2 + i
    # C00 (k'=0, i=0) idx 0
    # C01 (k'=1, i=0) idx 2
    # C10 (k'=0, i=1) idx 1
    # C11 (k'=1, i=1) idx 3
    W_coeffs = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, C11
        [0, 0, 1, 1], # M3: C01, C11
        [1, 1, 0, 0], # M4: C00, C10
        [-1, 0, 1, 0],# M5: C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 1]  # M7: C00, C11
    ]
    
    num_blocks = p // 2 # 14 // 2 = 7
    
    for r in range(num_blocks):
        for t in range(7):
            c = r * 7 + t
            # Fill U
            for i in range(2):
                for j in range(2):
                    idx_u = i * m + j
                    val = U_coeffs[t][i * 2 + j]
                    if val != 0:
                        U[idx_u][c] = val
                        
            # Fill V
            for j in range(2):
                for kp in range(2):
                    idx_v = j * p + (2 * r + kp)
                    val = V_coeffs[t][j * 2 + kp]
                    if val != 0:
                        V[idx_v][c] = val
                        
            # Fill W
            for kp in range(2):
                for i in range(2):
                    idx_w = (2 * r + kp) * n + i
                    val = W_coeffs[t][kp * 2 + i]
                    if val != 0:
                        W[idx_w][c] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
