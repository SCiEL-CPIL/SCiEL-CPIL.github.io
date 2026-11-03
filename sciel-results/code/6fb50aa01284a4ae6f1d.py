"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    # 6 sub-problems: (j_base, k_base)
    # A is 2x4, split into left (cols 0-1) and right (cols 2-3)
    # B is 4x6, split into top (rows 0-1) and bottom (rows 2-3)
    # 6 sub-problems: 2 (left/right) x 3 (col groups of B)
    subproblems = []
    for a_block in range(2):  # 0=left, 1=right
        j_base = a_block * 2
        for k_group in range(3):
            k_base = k_group * 2
            subproblems.append((j_base, k_base))
    
    rank = 6 * 7  # 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    
    # U coefficients for each M_t: dict from (i, j_local) to coeff
    U_coeffs = {
        0: {(0,0): 1, (1,1): 1},  # M1
        1: {(1,0): 1, (1,1): 1},  # M2
        2: {(0,0): 1},            # M3
        3: {(1,1): 1},            # M4
        4: {(0,0): 1, (0,1): 1},  # M5
        5: {(1,0): 1, (0,0): -1}, # M6
        6: {(0,1): 1, (1,1): -1}, # M7
    }
    
    # V coefficients for each M_t: dict from (j_local, k_local) to coeff
    V_coeffs = {
        0: {(0,0): 1, (1,1): 1},  # M1
        1: {(0,0): 1},            # M2
        2: {(0,1): 1, (1,1): -1}, # M3
        3: {(1,0): 1, (0,0): -1}, # M4
        4: {(1,1): 1},            # M5
        5: {(0,0): 1, (0,1): 1},  # M6
        6: {(1,0): 1, (1,1): 1},  # M7
    }
    
    # W coefficients: for each M_t, contribution to c[i,k_local]
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    W_coeffs = {
        0: {(0,0): 1, (1,1): 1},  # M1 -> c00+=1, c11+=1
        1: {(1,0): 1, (1,1): -1}, # M2 -> c10+=1, c11-=1
        2: {(0,1): 1, (1,1): 1},  # M3 -> c01+=1, c11+=1
        3: {(0,0): 1, (1,0): 1},  # M4 -> c00+=1, c10+=1
        4: {(0,0): -1, (0,1): 1}, # M5 -> c00-=1, c01+=1
        5: {(1,1): 1},            # M6 -> c11+=1
        6: {(0,0): 1},            # M7 -> c00+=1
    }
    
    for s, (j_base, k_base) in enumerate(subproblems):
        for t in range(7):
            col = s * 7 + t
            # Fill U
            for (i, j_local), coeff in U_coeffs[t].items():
                row = i * m + (j_base + j_local)
                U[row][col] += coeff
            # Fill V
            for (j_local, k_local), coeff in V_coeffs[t].items():
                row = (j_base + j_local) * p + (k_base + k_local)
                V[row][col] += coeff
            # Fill W
            for (k_local, i), coeff in W_coeffs[t].items():
                row = (k_base + k_local) * n + i
                W[row][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
