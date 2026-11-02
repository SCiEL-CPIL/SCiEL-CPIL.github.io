import time
# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49

    # Strassen 2x2 formulas
    strassen_2x2 = [
        # M1 = (a+d)(e+h)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (c+d)*e
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = a*(f-h)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = d*(g-e)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (a+b)*h
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (c-a)*(e+f)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (b-d)*(g+h)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Block-level Strassen for 4x4 (same structure as 2x2)
    block_strassen = strassen_2x2

    # Pre-calculate indices for speed and clarity
    # For each term t and s, we have a rank-1 component.
    # We can pre-calculate the (row, col) indices for U, V, W for each (t, s) pair.
    
    # However, the current loop is fast enough.
    # We just ensure correctness.

    for t in range(7):
        block_A, block_B, block_C = block_strassen[t]
        for s in range(7):
            scalar_A, scalar_B, scalar_C = strassen_2x2[s]
            r_idx = t * 7 + s

            # U
            for (bi, bj), bc in block_A.items():
                for (sr, sc), sc_coeff in scalar_A.items():
                    a_row = 2 * bi + sr
                    a_col = 2 * bj + sc
                    U[a_row * 4 + a_col][r_idx] += bc * sc_coeff

            # V
            for (bi, bj), bc in block_B.items():
                for (sr, sc), sc_coeff in scalar_B.items():
                    b_row = 2 * bi + sr
                    b_col = 2 * bj + sc
                    V[b_row * 4 + b_col][r_idx] += bc * sc_coeff

            # W
            for (ci, cj), bc in block_C.items():
                for (sr, sc), sc_coeff in scalar_C.items():
                    c_row = 2 * ci + sr
                    c_col = 2 * cj + sc
                    # W[k*4+i] where k=c_col, i=c_row
                    W[c_col * 4 + c_row][r_idx] += bc * sc_coeff

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
