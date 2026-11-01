import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps definition
    # Each step is defined by:
    # U_coeffs: { (i, j): coeff } for A
    # V_coeffs: { (j, k): coeff } for B (local block indices)
    # W_coeffs: { (i, k): coeff } for C (local block indices)
    
    # Step 1: M1 = (a00+a11)(b00+b11)
    # C00 += M1, C11 += M1
    step1 = {
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    }
    # Step 2: M2 = a11(b00+b11)
    # C00 -= M2
    step2 = {
        'U': {(1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): -1}
    }
    # Step 3: M3 = (a00+a01)b11
    # C01 += M3
    step3 = {
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1}
    }
    # Step 4: M4 = a00(b01-b11)
    # C00 += M4, C01 += M4, C11 -= M4
    step4 = {
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,0): 1, (0,1): 1, (1,1): -1}
    }
    # Step 5: M5 = (a10+a00)b00
    # C10 += M5
    step5 = {
        'U': {(1,0): 1, (0,0): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    }
    # Step 6: M6 = (a01-a10)(b00+b10)
    # C10 += M6
    step6 = {
        'U': {(0,1): 1, (1,0): -1},
        'V': {(0,0): 1, (1,0): 1},
        'W': {(1,0): 1}
    }
    # Step 7: M7 = (a10+a11)(b11-b00)
    # C11 += M7
    step7 = {
        'U': {(1,0): 1, (1,1): 1},
        'V': {(1,1): 1, (0,0): -1},
        'W': {(1,1): 1}
    }
    
    steps = [step1, step2, step3, step4, step5, step6, step7]
    
    # Part 1: A1 (cols 0,1) * B1 (rows 0,1)
    # A1 local (i,j) -> Global (i, j)
    # B1 local (j,k) -> Global (j, k) where j is 0,1 and k is 0..5
    # B1 is split into 3 blocks of 2 cols: k in {0,1}, {2,3}, {4,5}
    
    for t in range(3):
        for s_idx in range(7):
            c = t * 7 + s_idx
            step = steps[s_idx]
            
            # Fill U
            for (li, lj), val in step['U'].items():
                # li is local row (0,1), lj is local col (0,1)
                # Global A index: row i=li, col j=lj
                # U index: i*m + j
                u_idx = li * m + lj
                U[u_idx][c] += val
                
            # Fill V
            for (lj, lk), val in step['V'].items():
                # lj is local row (0,1) -> Global row j = lj
                # lk is local col (0,1) -> Global col k = 2*t + lk
                # V index: j*p + k
                v_idx = lj * p + (2 * t + lk)
                V[v_idx][c] += val
                
            # Fill W
            for (li, lk), val in step['W'].items():
                # li is local row (0,1) -> Global row i = li
                # lk is local col (0,1) -> Global col k = 2*t + lk
                # W index: k*n + i
                w_idx = (2 * t + lk) * n + li
                W[w_idx][c] += val
                
    # Part 2: A2 (cols 2,3) * B2 (rows 2,3)
    # A2 local (i,j) -> Global (i, j+2)
    # B2 local (j,k) -> Global (j+2, k) where j is 0,1 and k is 0..5
    # B2 is split into 3 blocks of 2 cols: k in {0,1}, {2,3}, {4,5}
    
    for t in range(3):
        for s_idx in range(7):
            c = 21 + t * 7 + s_idx
            step = steps[s_idx]
            
            # Fill U
            for (li, lj), val in step['U'].items():
                # Global A index: row i=li, col j=lj+2
                u_idx = li * m + (lj + 2)
                U[u_idx][c] += val
                
            # Fill V
            for (lj, lk), val in step['V'].items():
                # Global V index: row j = lj+2, col k = 2*t + lk
                v_idx = (lj + 2) * p + (2 * t + lk)
                V[v_idx][c] += val
                
            # Fill W
            for (li, lk), val in step['W'].items():
                # Global W index: row k = 2*t + lk, col i = li
                w_idx = (2 * t + lk) * n + li
                W[w_idx][c] += val
                
    return {"U": U, "V": V, "W": W}
