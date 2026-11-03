"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's 7 terms for 2x2x2
    # Each term is defined by coefficients for A (4 entries), B (4 entries), C (4 entries)
    # Indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    # A indices: i*m + j. For 2x2: 0, 1, 2, 3.
    # B indices: j*p + k. For 2x2: 0, 1, 2, 3.
    # C indices: k*n + i. For 2x2: 0, 1, 2, 3.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # Coefficients for A part (flattened 2x2)
    # 0:00, 1:01, 2:10, 3:11
    A_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [1, 0, -1, 0], # M6 (A10 - A00) -> index 2 is 10, index 0 is 00. Wait.
                       # Flattened A: [00, 01, 10, 11] -> [0, 1, 2, 3]
                       # A10 is index 2. A00 is index 0.
                       # So [ -1, 0, 1, 0 ]
        [0, 1, 0, -1]  # M7 (A01 - A11) -> index 1 is 01, index 3 is 11.
                       # So [ 0, 1, 0, -1 ]
    ]
    # Correcting M6 A_coeffs:
    A_coeffs[5] = [-1, 0, 1, 0]
    
    # Coefficients for B part (flattened 2x2)
    # 0:00, 1:01, 2:10, 3:11
    B_coeffs = [
        [1, 0, 0, 1],  # M1 (B00+B11)
        [1, 0, 0, 0],  # M2 (B00)
        [0, 0, 1, -1], # M3 (B10-B11)
        [0, 1, 0, -1], # M4 (B01-B00)
        [0, 0, 0, 1],  # M5 (B11)
        [1, 0, 1, 0],  # M6 (B00+B10)
        [0, 1, 0, 1]   # M7 (B11+B01)
    ]
    
    # Coefficients for C part (flattened 2x2)
    # 0:00, 1:01, 2:10, 3:11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [1, 0, 0, 0],  # M1 contributes to C00
        [0, 0, 0, 0],  # M2 contributes to C10
        [0, 0, 0, 0],  # M3 contributes to C01, C11
        [0, 0, 0, 0],  # M4 contributes to C00, C10
        [0, 0, 0, 0],  # M5 contributes to C00, C01
        [0, 0, 0, 0],  # M6 contributes to C11
        [0, 0, 0, 0]   # M7 contributes to C00
    ]
    # This is getting messy. Let's just build the C contribution per term directly in the loop.
    # C term for each Strassen step s:
    # s=0 (M1): C00 += 1, C11 += 1
    # s=1 (M2): C10 += 1
    # s=2 (M3): C01 += 1, C11 += 1
    # s=3 (M4): C00 += 1, C10 += 1
    # s=4 (M5): C00 -= 1, C01 += 1
    # s=5 (M6): C11 += 1
    # s=6 (M7): C00 += 1
    
    C_contrib = [
        {0: 1, 3: 1},      # M1
        {2: 1},            # M2
        {1: 1, 3: 1},      # M3
        {0: 1, 2: 1},      # M4
        {0: -1, 1: 1},     # M5
        {3: 1},            # M6
        {0: 1}             # M7
    ]

    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    # Helper to add a Strassen block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # i0, j0, k0 are the starting indices for the 2x2 block
        for s in range(7):
            # A coeffs
            for li in range(2):
                for lj in range(2):
                    coeff = A_coeffs[s][li * 2 + lj]
                    if coeff != 0:
                        gi = i0 + li
                        gj = j0 + lj
                        U[gi * m + gj][col] = coeff
            
            # B coeffs
            for lj in range(2):
                for lk in range(2):
                    coeff = B_coeffs[s][lj * 2 + lk]
                    if coeff != 0:
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = coeff
            
            # C coeffs
            for lk in range(2):
                for li in range(2):
                    # C index is k*n + i
                    # Local C index is lk*2 + li
                    lci = lk * 2 + li
                    if lci in C_contrib[s]:
                        coeff = C_contrib[s][lci]
                        gk = k0 + lk
                        gi = i0 + li
                        W[gk * n + gi][col] = coeff
            
            col += 1

    # Block 1: I{0,1}, J{0,1}, K{0,1}
    add_strassen(0, 0, 0)
    
    # Block 2: I{0,1}, J{0,1}, K{2,3}
    add_strassen(0, 0, 2)
    
    # Block 3: I{2}, J{0,1}, K{0,1}
    for j in [0, 1]:
        for k in [0, 1]:
            add_schoolbook(2, j, k)
            
    # Block 4: I{2}, J{0,1}, K{2,3}
    for j in [0, 1]:
        for k in [2, 3]:
            add_schoolbook(2, j, k)
            
    # Block 5: I{0,1}, J{2}, K{0,1}
    for i in [0, 1]:
        for k in [0, 1]:
            add_schoolbook(i, 2, k)
            
    # Block 6: I{0,1}, J{2}, K{2,3}
    for i in [0, 1]:
        for k in [2, 3]:
            add_schoolbook(i, 2, k)
            
    # Block 7: I{2}, J{2}, K{0,1}
    for k in [0, 1]:
        add_schoolbook(2, 2, k)
        
    # Block 8: I{2}, J{2}, K{2,3}
    for k in [2, 3]:
        add_schoolbook(2, 2, k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
