"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # (2*4, 42) = (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (4*6, 42) = (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (6*2, 42) = (12, 42)
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # For matrices X (2x2) and Y (2x2), product Z = XY:
    # P1 = (X00 + X11) * (Y00 + Y11)
    # P2 = (X10 + X11) * Y00
    # P3 = X00 * (Y01 - Y11)
    # P4 = X11 * (Y10 - Y00)
    # P5 = (X00 + X01) * Y11
    # P6 = (X10 - X00) * (Y00 + Y01)
    # P7 = (X01 - X11) * (Y10 + Y11)
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # U coefficients: for each Strassen term s, dict mapping (i_local, j_local) -> coeff
    # i_local in {0,1}, j_local in {0,1}
    U_strassen = [
        {(0, 0): 1, (1, 1): 1},      # P1: X00 + X11
        {(1, 0): 1, (1, 1): 1},      # P2: X10 + X11
        {(0, 0): 1},                  # P3: X00
        {(1, 1): 1},                  # P4: X11
        {(0, 0): 1, (0, 1): 1},      # P5: X00 + X01
        {(1, 0): 1, (0, 0): -1},     # P6: X10 - X00
        {(0, 1): 1, (1, 1): -1}      # P7: X01 - X11
    ]
    
    # V coefficients: for each Strassen term s, dict mapping (j_local, k_local) -> coeff
    V_strassen = [
        {(0, 0): 1, (1, 1): 1},      # P1: Y00 + Y11
        {(0, 0): 1},                  # P2: Y00
        {(0, 1): 1, (1, 1): -1},     # P3: Y01 - Y11
        {(1, 0): 1, (0, 0): -1},     # P4: Y10 - Y00
        {(1, 1): 1},                  # P5: Y11
        {(0, 0): 1, (0, 1): 1},      # P6: Y00 + Y01
        {(1, 0): 1, (1, 1): 1}       # P7: Y10 + Y11
    ]
    
    # W coefficients: for each Strassen term s, dict mapping (k_local, i_local) -> coeff
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    W_strassen = [
        {(0, 0): 1, (1, 1): 1},      # P1: contributes to Z00(+1), Z11(+1)
        {(0, 1): 1, (1, 1): -1},     # P2: contributes to Z10(+1), Z11(-1)
        {(1, 0): 1, (1, 1): 1},      # P3: contributes to Z01(+1), Z11(+1)
        {(0, 0): 1, (0, 1): 1},      # P4: contributes to Z00(+1), Z10(+1)
        {(0, 0): -1, (1, 0): 1},     # P5: contributes to Z00(-1), Z01(+1)
        {(1, 1): 1},                 # P6: contributes to Z11(+1)
        {(0, 0): 1}                  # P7: contributes to Z00(+1)
    ]
    
    col = 0
    # j_groups: split the 4 columns of A (and rows of B) into 2 groups of 2
    j_groups = [(0, 1), (2, 3)]
    # k_groups: split the 6 columns of B into 3 groups of 2
    k_groups = [(0, 1), (2, 3), (4, 5)]
    
    for jg in j_groups:
        j_base = jg[0]  # 0 or 2
        for kg in k_groups:
            k_base = kg[0]  # 0, 2, or 4
            for s in range(7):
                # Update U: indices (i, j) where j = j_base + j_local
                for (i_loc, j_loc), coeff in U_strassen[s].items():
                    i = i_loc
                    j = j_base + j_loc
                    u_idx = i * m + j  # i*4 + j
                    U[u_idx][col] += coeff
                
                # Update V: indices (j, k) where j = j_base + j_local, k = k_base + k_local
                for (j_loc, k_loc), coeff in V_strassen[s].items():
                    j = j_base + j_loc
                    k = k_base + k_loc
                    v_idx = j * p + k  # j*6 + k
                    V[v_idx][col] += coeff
                
                # Update W: indices (k, i) where k = k_base + k_local, i = i_local
                for (k_loc, i_loc), coeff in W_strassen[s].items():
                    k = k_base + k_loc
                    i = i_loc
                    w_idx = k * n + i  # k*2 + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
