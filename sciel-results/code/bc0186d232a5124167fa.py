"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen algorithm for 2x2 blocks of 3x3 matrices
    # Total rank: 7 * 27 = 189
    
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps
    # Each step is a tuple (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {(alpha, beta): coeff}
    # B_coeffs: dict {(beta, gamma): coeff}
    # C_coeffs: dict {(alpha, gamma): coeff}
    
    steps = []
    
    # M1 = (A00 + A11)(B00 + B11)
    steps.append({
        'A': {(0,0): 1, (1,1): 1},
        'B': {(0,0): 1, (0,1): 1}, # Wait, B00 is (0,0), B11 is (1,1)?
        # B indices: (row_block, col_block).
        # B00: row 0, col 0. B11: row 1, col 1.
        # So B00 is (0,0), B11 is (1,1).
        # My previous dict keys for B were (beta, gamma).
        # beta is row block of B (shared with A col). gamma is col block of B.
        # So B00 is (0,0). B11 is (1,1).
        'C': {(0,0): 1, (1,1): 1} # C00 gets M1, C11 gets M1
    })
    # Correction:
    # M1 = (A00 + A11)(B00 + B11)
    # A00: (0,0). A11: (1,1).
    # B00: (0,0). B11: (1,1).
    # C00 = M1 + ...
    # C11 = M1 - ...
    
    steps[0] = {
        'A': {(0,0): 1, (1,1): 1},
        'B': {(0,0): 1, (1,1): 1},
        'C': {(0,0): 1, (1,1): 1}
    }
    
    # M2 = (A10 + A11)B00
    steps.append({
        'A': {(1,0): 1, (1,1): 1},
        'B': {(0,0): 1},
        'C': {(1,0): 1, (1,1): -1} # C10 = M2 + ..., C11 = ... - M2
    })
    
    # M3 = A00(B01 - B11)
    steps.append({
        'A': {(0,0): 1},
        'B': {(0,1): 1, (1,1): -1}, # B01 is (0,1), B11 is (1,1)
        'C': {(0,1): 1, (1,1): 1} # C01 = M3 + ..., C11 = ... + M3
    })
    
    # M4 = A11(B10 - B00)
    steps.append({
        'A': {(1,1): 1},
        'B': {(1,0): 1, (0,0): -1}, # B10 is (1,0), B00 is (0,0)
        'C': {(0,0): 1, (1,0): 1} # C00 = ... + M4, C10 = ... + M4
    })
    
    # M5 = (A00 + A01)B11
    steps.append({
        'A': {(0,0): 1, (0,1): 1},
        'B': {(1,1): 1},
        'C': {(0,0): -1, (0,1): 1} # C00 = ... - M5, C01 = ... + M5
    })
    
    # M6 = (A10 - A00)(B00 + B01)
    steps.append({
        'A': {(1,0): 1, (0,0): -1},
        'B': {(0,0): 1, (0,1): 1},
        'C': {(1,1): 1} # C11 = ... + M6
    })
    
    # M7 = (A01 - A11)(B10 + B11)
    steps.append({
        'A': {(0,1): 1, (1,1): -1},
        'B': {(1,0): 1, (1,1): 1},
        'C': {(0,0): 1} # C00 = ... + M7
    })
    
    col = 0
    for s in steps:
        A_coeffs = s['A']
        B_coeffs = s['B']
        C_coeffs = s['C']
        
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # U part
                    for (a, b), coeff_a in A_coeffs.items():
                        i = a * 3 + u
                        j = b * 3 + w
                        U[i * 6 + j][col] += coeff_a
                    
                    # V part
                    for (b, g), coeff_b in B_coeffs.items():
                        j = b * 3 + w
                        k = g * 3 + v
                        V[j * 6 + k][col] += coeff_b
                    
                    # W part
                    for (a, g), coeff_c in C_coeffs.items():
                        k = g * 3 + v
                        i = a * 3 + u
                        W[k * 6 + i][col] += coeff_c
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
