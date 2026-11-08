import time

def construct():
    start_time = time.time()
    deadline = 150  # seconds
    
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42 (i, j) -> i is row of A (0..1), j is col of A (0..3)
    # Wait, U shape is (n*m, rank). n=2, m=4. Index is i*m + j.
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42 (j, k) -> j is row of B (0..3), k is col of B (0..5)
    # Index is j*p + k.
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42 (k, i) -> k is col of B (0..5), i is row of A (0..1)
    # Index is k*n + i.

    # Strassen coefficients for 2x2 multiplication
    # U_coeffs[s]: dict mapping (i_local, j_local) -> coeff for A block
    # V_coeffs[s]: dict mapping (j_local, k_local) -> coeff for B block
    # W_coeffs[s]: dict mapping (i_local, k_local) -> coeff for C block
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # M1: (A00 + A11)
        {(1, 0): 1, (1, 1): 1},       # M2: (A10 + A11)
        {(0, 0): 1},                   # M3: A00
        {(1, 1): 1},                   # M4: A11
        {(0, 0): 1, (0, 1): 1},       # M5: (A00 + A01)
        {(1, 0): 1, (0, 0): -1},      # M6: (A10 - A00)
        {(0, 1): 1, (1, 1): -1}       # M7: (A01 - A11)
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # M1: (B00 + B11)
        {(0, 0): 1},                   # M2: B00
        {(0, 1): 1, (1, 1): -1},      # M3: (B01 - B11)
        {(1, 0): 1, (0, 0): -1},      # M4: (B10 - B00)
        {(1, 1): 1},                   # M5: B11
        {(0, 0): 1, (0, 1): 1},       # M6: (B00 + B01)
        {(1, 0): 1, (1, 1): 1}        # M7: (B10 + B11)
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # M1: C00 += 1, C11 += 1
        {(1, 0): 1, (1, 1): -1},      # M2: C10 += 1, C11 += -1
        {(0, 1): 1, (1, 1): 1},       # M3: C01 += 1, C11 += 1
        {(0, 0): 1, (1, 0): 1},       # M4: C00 += 1, C10 += 1
        {(0, 0): -1, (0, 1): 1},      # M5: C00 += -1, C01 += 1
        {(1, 1): 1},                   # M6: C11 += 1
        {(0, 0): 1}                    # M7: C00 += 1
    ]

    # side: 0 for A_L (cols 0,1), 1 for A_R (cols 2,3)
    # j_block: 0, 1, 2 for B col blocks (cols 0-1, 2-3, 4-5)
    # s: 0..6 for Strassen term
    
    r = 0
    for side in range(2):
        for j_block in range(3):
            for s in range(7):
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = side * 2 + j_loc
                    U[i * m + j][r] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = side * 2 + j_loc
                    k = j_block * 2 + k_loc
                    V[j * p + k][r] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = j_block * 2 + k_loc
                    W[k * n + i][r] += coeff
                
                r += 1
                
                if time.time() > deadline - 5:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
