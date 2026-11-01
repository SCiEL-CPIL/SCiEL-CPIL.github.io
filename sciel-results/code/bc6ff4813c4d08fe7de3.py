import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 blocks * 7 Strassen terms = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for one 2x2 block
    # Indices for U (A): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for V (B): 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # Indices for W (C): 00, 01, 10, 11 -> 0, 1, 2, 3 (local, k*n+i)
    # Local k: 0, 1. Local i: 0, 1.
    # W index local: k*2 + i.
    
    # M1 = a(f-g)
    # M2 = (a+b)g
    # M3 = (c+d)e
    # M4 = d(h-e)
    # M5 = (a+d)(e+h)
    # M6 = (b-d)(f+h)
    # M7 = (c-a)(e+g)
    
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 - M3 + M5 + M7
    
    # Coeffs: [M1, M2, M3, M4, M5, M6, M7]
    # U coeffs (for A_00, A_01, A_10, A_11)
    U_coeffs = [
        [1, 0, 0, 0], # M1: a
        [1, 1, 0, 0], # M2: a+b
        [0, 1, 1, 0], # M3: c+d (Wait, c=A10, d=A11) -> [0, 0, 1, 1]
        [0, 0, 0, 1], # M4: d
        [1, 0, 0, 1], # M5: a+d
        [0, 1, 0, -1],# M6: b-d
        [0, 0, -1, 1] # M7: c-a (Wait, c=A10, a=A00) -> [ -1, 0, 1, 0 ]? No.
                      # M7 = (c-a). c=A10 (index 2), a=A00 (index 0). So [-1, 0, 1, 0].
    ]
    # Correction M3: (c+d). c=A10, d=A11. Indices 2, 3. [0, 0, 1, 1].
    # Correction M7: (c-a). c=A10, a=A00. Indices 2, 0. [-1, 0, 1, 0].
    
    U_coeffs = [
        [1, 0, 0, 0], # M1
        [1, 1, 0, 0], # M2
        [0, 0, 1, 1], # M3
        [0, 0, 0, 1], # M4
        [1, 0, 0, 1], # M5
        [0, 1, 0, -1],# M6
        [-1, 0, 1, 0] # M7
    ]
    
    # V coeffs (for B_00, B_01, B_10, B_11)
    # B indices: 00(0), 01(1), 10(2), 11(3)
    V_coeffs = [
        [0, 1, -1, 0], # M1: f-g
        [0, 0, 1, 0],   # M2: g
        [1, 0, 0, 0],   # M3: e
        [1, -1, 0, 0],  # M4: h-e? No, M4 = d(h-e). h=B11, e=B00. [0, 0, 0, 1] for d? 
                        # M4 = d(h-e). V part is h-e. h=B11(3), e=B00(0). [ -1, 0, 0, 1 ].
        [1, 0, 0, 1],   # M5: e+h
        [0, 1, 0, 1],   # M6: f+h
        [1, 0, 1, 0]    # M7: e+g
    ]
    # Correction M4: h-e. h=B11, e=B00. [-1, 0, 0, 1].
    
    V_coeffs = [
        [0, 1, -1, 0], # M1
        [0, 0, 1, 0],   # M2
        [1, 0, 0, 0],   # M3
        [-1, 0, 0, 1],  # M4
        [1, 0, 0, 1],   # M5
        [0, 1, 0, 1],   # M6
        [1, 0, 1, 0]    # M7
    ]
    
    # W coeffs (for C_00, C_01, C_10, C_11)
    # C indices local: 00(0), 01(1), 10(2), 11(3)
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 - M3 + M5 + M7
    
    W_coeffs = [
        [0, 1, 0, 1], # M1: C01(1), C11(1)
        [-1, 1, 0, 0],# M2: C00(-1), C01(1)
        [0, 0, 1, -1],# M3: C10(1), C11(-1)
        [1, 0, 1, 0], # M4: C00(1), C10(1)
        [1, 0, 0, 1], # M5: C00(1), C11(1)
        [1, 0, 0, 0], # M6: C00(1)
        [0, 0, 0, 1]  # M7: C11(1)
    ]
    
    for l in range(8):
        base_col = l * 7
        k0 = 2 * l
        k1 = 2 * l + 1
        
        for r in range(7):
            col = base_col + r
            
            # U
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j
                    U[u_idx][col] = U_coeffs[r][i * 2 + j]
            
            # V
            for j in range(2):
                # k=0 -> k0, k=1 -> k1
                v_k0 = V_coeffs[r][j * 2 + 0]
                v_k1 = V_coeffs[r][j * 2 + 1]
                V[j * p + k0][col] = v_k0
                V[j * p + k1][col] = v_k1
            
            # W
            # W index is k * n + i
            # C00: k=0, i=0 -> 0
            # C01: k=0, i=1 -> 1
            # C10: k=1, i=0 -> 2
            # C11: k=1, i=1 -> 3
            w_vals = W_coeffs[r]
            W[k0 * n + 0][col] = w_vals[0]
            W[k0 * n + 1][col] = w_vals[1]
            W[k1 * n + 0][col] = w_vals[2]
            W[k1 * n + 1][col] = w_vals[3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
