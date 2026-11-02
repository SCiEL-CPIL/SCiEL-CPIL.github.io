"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p is 12, so there are 6 blocks of size 2
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict (i, j) -> coeff
    # V_coeffs: dict (j, l) -> coeff (l is local col 0 or 1)
    # W_coeffs: dict (l, i) -> coeff (l is local col 0 or 1)
    
    # s=0: M1 = (A00+A11)(B00+B11)
    # C00 += 1, C11 += 1
    term_0 = (
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    )
    
    # s=1: M2 = (A10+A11)B00
    # C10 += 1, C11 += -1
    term_1 = (
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1}
    )
    
    # s=2: M3 = A00(B10-B11)
    # C01 += 1, C11 += 1
    term_2 = (
        {(0,0): 1},
        {(1,0): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1}
    )
    
    # s=3: M4 = A11(B01-B00)
    # C00 += 1, C10 += 1
    term_3 = (
        {(1,1): 1},
        {(0,1): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1}
    )
    
    # s=4: M5 = (A00+A10)B11
    # C00 += -1, C01 += 1
    term_4 = (
        {(0,0): 1, (1,0): 1},
        {(1,1): 1},
        {(0,0): -1, (1,0): 1}
    )
    
    # s=5: M6 = (A10-A00)(B00+B01)
    # C11 += 1
    term_5 = (
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    )
    
    # s=6: M7 = (A11-A01)(B10+B11)
    # C00 += 1
    term_6 = (
        {(1,1): 1, (0,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    )
    
    terms = [term_0, term_1, term_2, term_3, term_4, term_5, term_6]
    
    for b in range(num_blocks):
        for s in range(7):
            col = 7 * b + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for (i, j), val in u_coeffs.items():
                U[i * m + j][col] += val
                
            for (j, l), val in v_coeffs.items():
                k = 2 * b + l
                V[j * p + k][col] += val
                
            for (l, i), val in w_coeffs.items():
                k = 2 * b + l
                W[k * n + i][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
