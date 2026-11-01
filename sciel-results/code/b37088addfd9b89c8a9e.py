import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-28 exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    Uses Strassen's algorithm for each of the 4 blocks of 2x2 in the second matrix.
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 * 2x2
    # A = [[a, b], [c, d]]
    # B = [[x0, x1], [y0, y1]]
    # C = [[z0, z1], [z2, z3]]
    
    # M1 = (a+d)(x0+y1)
    # M2 = (c+d)x0
    # M3 = a(x1-y1)
    # M4 = d(y0-x0)
    # M5 = (a+b)y1
    # M6 = (c-a)(x0+x1)
    # M7 = (b-d)(y0+y1)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the linear forms for A and B for each M
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3 (local to block)
    
    A_forms = [
        {0: 1, 3: 1},       # M1: a+d
        {2: 1, 3: 1},       # M2: c+d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a+b
        {2: 1, 0: -1},      # M6: c-a
        {1: 1, 3: -1}       # M7: b-d
    ]
    
    B_forms = [
        {0: 1, 3: 1},       # M1: x0+y1
        {0: 1},             # M2: x0
        {1: 1, 3: -1},      # M3: x1-y1
        {2: 1, 0: -1},      # M4: y0-x0
        {3: 1},             # M5: y1
        {0: 1, 1: 1},       # M6: x0+x1
        {2: 1, 3: 1}        # M7: y0+y1
    ]
    
    # Coefficients for C
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # M1, M4, M5, M7
        {2: 1, 4: 1},              # M3, M5
        {1: 1, 3: 1},              # M2, M4
        {0: 1, 1: -1, 2: 1, 5: 1}  # M1, M2, M3, M6
    ]
    # Note: indices in C_coeffs correspond to M1..M7 (0..6)
    # C00 uses M1(0), M4(3), M5(4), M7(6)
    # C01 uses M3(2), M5(4)
    # C10 uses M2(1), M4(3)
    # C11 uses M1(0), M2(1), M3(2), M6(5)

    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        for s in range(7):
            r = block * 7 + s
            
            # Fill U
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            
            # Fill V
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            
            # Fill W
            # C_coeffs[c_idx] is a dict of {m_idx: coeff}
            for c_idx, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_idx // 2
                    k_local = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
