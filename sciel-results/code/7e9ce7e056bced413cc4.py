import numpy as np
import time

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds, with safety margin

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard: if we've exceeded our budget, return a valid (if not optimal) result
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Fallback: schoolbook rank 48 (always valid)
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
        return {"U": U, "V": V, "W": W}
    # Rank is 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A and B parts
    # M_s = sum_a coeff_a * A[i,j] * sum_b coeff_b * B[j,k]
    # We define the linear forms for A and B for each s
    
    # A forms: list of 7 dicts { (i, j_local): coeff }
    # B forms: list of 7 dicts { (j_local, k_local): coeff }
    
    # s=0: (A00 + A11) * (B00 + B11)
    # s=1: (A10 + A11) * B00
    # s=2: A00 * (B01 - B11)
    # s=3: A11 * (B10 - B00)
    # s=4: (A00 + A01) * B11
    # s=5: (A10 - A00) * (B00 + B01)
    # s=6: (A01 - A11) * (B10 + B11)
    
    A_forms = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    B_forms = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    # C reconstruction coefficients: C[i_local, k_local] = sum_s coeff[i_local, k_local, s] * M_s
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    C_coeffs = [
        [ # i_local=0
            [1, 0, 0, 1, -1, 0, 1], # k_local=0
            [0, 0, 1, 0, 1, 0, 0]   # k_local=1
        ],
        [ # i_local=1
            [0, 1, 0, 1, 0, 0, 0],  # k_local=0
            [1, -1, 1, 0, 0, 1, 0]  # k_local=1
        ]
    ]
    
    for b in range(6):
        # Determine A part and B part mapping
        # b 0,1,2 -> A1 (j 0,1), B1 (j 0,1)
        # b 3,4,5 -> A2 (j 2,3), B2 (j 2,3)
        
        if b < 3:
            j_offset_A = 0
            j_offset_B = 0
        else:
            j_offset_A = 2
            j_offset_B = 2
            
        k_offset = (b % 3) * 2
        
        for s in range(7):
            col = b * 7 + s
            
            # Fill U
            for (i_loc, j_loc), coeff in A_forms[s].items():
                i_global = i_loc
                j_global = j_loc + j_offset_A
                U[i_global * m + j_global][col] += coeff
                
            # Fill V
            for (j_loc, k_loc), coeff in B_forms[s].items():
                j_global = j_loc + j_offset_B
                k_global = k_loc + k_offset
                V[j_global * p + k_global][col] += coeff
                
            # Fill W
            for i_loc in range(2):
                for k_loc in range(2):
                    coeff = C_coeffs[i_loc][k_loc][s]
                    if coeff != 0:
                        i_global = i_loc
                        k_global = k_loc + k_offset
                        # W index is k*n + i
                        W[k_global * n + i_global][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
