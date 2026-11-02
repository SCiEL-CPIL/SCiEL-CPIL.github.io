"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 via recursive Strassen
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Top-level blocks for 4x4
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B10], [B11, B21]] -> Wait, B indices are (row, col).
    # B00 is top-left, B01 is top-right, B10 is bottom-left, B11 is bottom-right.
    
    # Coefficients for A_m (sum of blocks A_IJ)
    # m=0 (M1): A00 + A11
    # m=1 (M2): A10 + A11
    # m=2 (M3): A00
    # m=3 (M4): A11
    # m=4 (M5): A00 + A01
    # m=5 (M6): A10 - A00
    # m=6 (M7): A01 - A11
    
    A_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    # Coefficients for B_m (sum of blocks B_JK)
    # m=0 (M1): B00 + B11
    # m=1 (M2): B00
    # m=2 (M3): B01 - B11
    # m=3 (M4): B10 - B00
    # m=4 (M5): B11
    # m=5 (M6): B00 + B01
    # m=6 (M7): B10 + B11
    
    B_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    # C combination coefficients
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C_IK is sum of M_m with coeff delta_IK_m
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}   # C11
    ]
    
    # Inner Strassen linear forms for A (coeffs for a00, a01, a10, a11)
    # Index order: (0,0), (0,1), (1,0), (1,1)
    A_inner = [
        [1, 0, 0, 1], # M1: a00 + a11
        [0, 0, 1, 1], # M2: a10 + a11
        [1, 0, 0, 0], # M3: a00
        [0, 0, 0, 1], # M4: a11
        [1, 1, 0, 0], # M5: a00 + a01
        [-1, 0, 1, 0],# M6: a10 - a00
        [0, 1, 0, -1] # M7: a01 - a11
    ]
    
    B_inner = [
        [1, 0, 0, 1], # M1: b00 + b11
        [1, 0, 0, 0], # M2: b00
        [0, 1, 0, -1],# M3: b01 - b11
        [0, 0, 1, -1],# M4: b10 - b00
        [0, 0, 0, 1], # M5: b11
        [1, 1, 0, 0], # M6: b00 + b01
        [0, 0, 1, 1]  # M7: b10 + b11
    ]
    
    # C_inner: T_m entries from S_k
    # T00 = S1 + S4 - S5 + S7
    # T01 = S3 + S5
    # T10 = S2 + S4
    # T11 = S1 - S2 + S3 + S6
    # Index order for T: (0,0), (0,1), (1,0), (1,1)
    # Index order for S: 0..6
    C_inner = [
        [1, 0, 0, 1, -1, 0, 1], # T00
        [0, 0, 1, 0, 1, 0, 0],   # T01
        [0, 1, 0, 1, 0, 0, 0],   # T10
        [1, -1, 1, 0, 0, 1, 0]   # T11
    ]
    
    col = 0
    for m in range(7):
        for k in range(7):
            # U coefficients
            for I in range(2):
                for J in range(2):
                    c_A = A_coeffs[m].get((I, J), 0)
                    if c_A == 0: continue
                    # a_pq in A_m comes from A_IJ
                    # A_inner[k] gives coeff for a_pq
                    for p in range(2):
                        for q in range(2):
                            c_inner = A_inner[k][p*2+q] # (p,q) -> p*2+q
                            if c_inner == 0: continue
                            global_i = 2*I + p
                            global_j = 2*J + q
                            U[global_i*m + global_j][col] += c_A * c_inner
                            
            # V coefficients
            for J in range(2):
                for K in range(2):
                    c_B = B_coeffs[m].get((J, K), 0)
                    if c_B == 0: continue
                    for p in range(2):
                        for q in range(2):
                            c_inner = B_inner[k][p*2+q]
                            if c_inner == 0: continue
                            global_j = 2*J + p
                            global_k = 2*K + q
                            V[global_j*p + global_k][col] += c_B * c_inner
                            
            # W coefficients
            for I in range(2):
                for K in range(2):
                    c_C = C_coeffs[I*2+K].get(m, 0)
                    if c_C == 0: continue
                    for p in range(2):
                        for q in range(2):
                            c_inner = C_inner[p*2+q][k]
                            if c_inner == 0: continue
                            global_i = 2*I + p
                            global_k = 2*K + q
                            W[global_k*n + global_i][col] += c_C * c_inner
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
