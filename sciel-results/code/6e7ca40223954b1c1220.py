import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (first matrix M, 2x2 local blocks)
    # Local index (i_loc, j_loc) -> global row i=i_loc, col j = a*2 + j_loc
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # P1: M00 + M11
        {(1, 0): 1, (1, 1): 1},   # P2: M10 + M11
        {(0, 0): 1},               # P3: M00
        {(1, 1): 1},               # P4: M11
        {(0, 0): 1, (0, 1): 1},   # P5: M00 + M01
        {(1, 0): 1, (0, 0): -1},  # P6: M10 - M00
        {(0, 1): 1, (1, 1): -1},  # P7: M01 - M11
    ]
    
    # Strassen coefficients for V (second matrix N, 2x2 local blocks)
    # Local index (j_loc, k_loc) -> global row j = a*2 + j_loc, col k = b*2 + k_loc
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # P1: N00 + N11
        {(0, 0): 1},               # P2: N00
        {(0, 1): 1, (1, 1): -1},  # P3: N01 - N11
        {(1, 0): 1, (0, 0): -1},  # P4: N10 - N00
        {(1, 1): 1},               # P5: N11
        {(0, 0): 1, (0, 1): 1},   # P6: N00 + N01
        {(1, 0): 1, (1, 1): 1},   # P7: N10 + N11
    ]
    
    # Strassen coefficients for W (output R, 2x2 local blocks)
    # Local index (i_loc, k_loc) -> global row k = b*2 + k_loc, col i = i_loc
    # Trace convention: W[k*n + i]
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},   # P1: R00 + R11
        {(1, 0): 1, (1, 1): -1},  # P2: R10 - R11
        {(0, 1): 1, (1, 1): 1},   # P3: R01 + R11
        {(0, 0): 1, (1, 0): 1},   # P4: R00 + R10
        {(0, 0): -1, (0, 1): 1},  # P5: -R00 + R01
        {(1, 1): 1},               # P6: R11
        {(0, 0): 1},               # P7: R00
    ]
    
    col = 0
    for a in range(2):  # A block: 0 for cols {0,1}, 1 for cols {2,3}
        for b in range(3):  # B col block: 0 for cols {0,1}, 1 for {2,3}, 2 for {4,5}
            for s in range(7):  # Strassen term
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
            
    return {"U": U, "V": V, "W": W}
