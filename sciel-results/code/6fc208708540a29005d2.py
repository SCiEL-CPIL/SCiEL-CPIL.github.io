import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L B_L using Strassen on 4 blocks
    # A_L is A[:, 0:2], B_L is B[0:2, :]
    # Blocks of B_L are columns 0-1, 2-3, 4-5, 6-7
    
    # Strassen terms definitions
    # M1: (A00+A11)(B00+B11) -> C00(-1), C10(1)
    # M2: (A10+A11)B00 -> C10(1), C11(0)? No, C10(1)
    # M3: A00(B01-B11) -> C11(1)
    # M4: A11(B10-B00) -> C00(-1), C01(1)
    # M5: (A00+A01)B11 -> C00(1), C10(-1)
    # M6: (A10-A00)(B00+B01) -> C10(-1), C11(1)
    # M7: (A01-A11)(B10+B11) -> C00(-1), C01(1)
    
    # Let's verify the C contributions again.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M4 + M7
    # C10 = M2 - M6 + M1 - M5
    # C11 = M3 + M6
    
    # For each block q (0..3):
    for q in range(4):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # B indices
        b00_idx = 0 * p + k0
        b01_idx = 0 * p + k1
        b10_idx = 1 * p + k0
        b11_idx = 1 * p + k1
        
        # A indices
        a00_idx = 0 * m + 0
        a01_idx = 0 * m + 1
        a10_idx = 1 * m + 0
        a11_idx = 1 * m + 1
        
        # C indices (W)
        # C00: i=0, k=k0 -> k0*n + 0
        # C01: i=0, k=k1 -> k1*n + 0
        # C10: i=1, k=k0 -> k0*n + 1
        # C11: i=1, k=k1 -> k1*n + 1
        c00_idx = k0 * n + 0
        c01_idx = k1 * n + 0
        c10_idx = k0 * n + 1
        c11_idx = k1 * n + 1
        
        # Term 1: M1
        U[a00_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c10_idx][col] += 1
        col += 1
        
        # Term 2: M2
        U[a10_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # Term 3: M3
        U[a00_idx][col] += 1
        V[b01_idx][col] += 1
        V[b11_idx][col] -= 1
        W[c11_idx][col] += 1
        col += 1
        
        # Term 4: M4
        U[a11_idx][col] += 1
        V[b10_idx][col] += 1
        V[b00_idx][col] -= 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # Term 5: M5
        U[a00_idx][col] += 1
        U[a01_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        W[c10_idx][col] -= 1
        col += 1
        
        # Term 6: M6
        U[a10_idx][col] += 1
        U[a00_idx][col] -= 1
        V[b00_idx][col] += 1
        V[b01_idx][col] += 1
        W[c10_idx][col] -= 1
        W[c11_idx][col] += 1
        col += 1
        
        # Term 7: M7
        U[a01_idx][col] += 1
        U[a11_idx][col] -= 1
        V[b10_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1

    # Part 2: A_R B_R (16 terms)
    # A_R is A[:, 2], B_R is B[2, :]
    for i in range(n):
        for k in range(p):
            a_idx = i * m + 2
            b_idx = 2 * p + k
            c_idx = k * n + i
            
            U[a_idx][col] += 1
            V[b_idx][col] += 1
            W[c_idx][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
