"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 16 x 28
    
    # Strassen for 2x2x2
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B local indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C local indices (W convention k*n+i): 0=c00, 1=c10, 2=c01, 3=c11
    
    U_coeffs = [
        {0: 1, 3: 1},       # M0: a00+a11
        {2: 1, 3: 1},       # M1: a10+a11
        {0: 1},              # M2: a00
        {3: 1},              # M3: a11
        {0: 1, 1: 1},       # M4: a00+a01
        {2: 1, 0: -1},      # M5: a10-a00
        {1: 1, 3: -1}       # M6: a01-a11
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},       # M0: b00+b11
        {0: 1},              # M1: b00
        {2: 1, 3: -1},      # M2: b10-b11
        {1: 1, 0: -1},      # M3: b01-b00
        {3: 1},              # M4: b11
        {0: 1, 2: 1},       # M5: b00+b10
        {1: 1, 3: 1}        # M6: b01+b11
    ]
    
    # W local: 0=c00(i=0,k=0), 1=c10(i=1,k=0), 2=c01(i=0,k=1), 3=c11(i=1,k=1)
    W_coeffs = [
        {0: 1, 3: 1},        # M0: c00+1, c11+1
        {1: 1},              # M1: c10+1
        {3: -1},             # M2: c11-1
        {0: -1, 1: 1},       # M3: c00-1, c10+1
        {0: -1, 2: 1},       # M4: c00-1, c01+1
        {3: -1},             # M5: c11-1
        {2: 1}               # M6: c01+1
    ]
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
                
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2       # row of B (0 or 1)
                k_local = v_local_idx % 2   # local col (0 or 1)
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
                
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2         # row of C (0 or 1)
                k_local = w_local_idx // 2  # local col (0 or 1)
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
