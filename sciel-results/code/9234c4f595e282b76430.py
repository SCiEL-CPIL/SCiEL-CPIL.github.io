"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    # Rank 28 using 4 Strassen blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2) and B (2x2)
    # A indices: 00, 01, 10, 11 -> U rows 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> V rows 0, 1, 8, 9 (for a specific block)
    # C indices: 00, 01, 10, 11 -> W rows (block dependent)
    
    # 7 Strassen products
    # m1: (a00+a11)(b00+b11)
    # m2: (a10+a11)b00
    # m3: a00(b10-b11)
    # m4: a11(b00-b01)
    # m5: (a00+a01)b11
    # m6: (a10-a00)(b00+b01)
    # m7: (a01-a11)(b10+b11)
    
    # C reconstruction:
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    strassen_A = [
        [1, 0, 0, 1], # m1: a00, a11
        [0, 0, 1, 1], # m2: a10, a11
        [1, 0, 0, 0], # m3: a00
        [0, 0, 0, 1], # m4: a11
        [1, 1, 0, 0], # m5: a00, a01
        [-1, 0, 1, 0], # m6: a10, a00
        [0, 1, 0, -1]  # m7: a01, a11
    ]
    
    strassen_B = [
        [1, 1, 0, 0], # m1: b00, b11 (indices 0, 3 in local 2x2)
        [1, 0, 0, 0], # m2: b00
        [0, 0, 1, -1], # m3: b10, b11
        [1, -1, 0, 0], # m4: b00, b01
        [0, 0, 0, 1], # m5: b11
        [1, 1, 0, 0], # m6: b00, b01
        [0, 0, 1, 1]  # m7: b10, b11
    ]
    
    # Local C indices in 2x2: 00, 01, 10, 11
    # W coefficients for each m to local C
    strassen_W = [
        [1, 0, 0, 1], # m1 -> c00, c11
        [0, 0, 1, -1], # m2 -> c10, c11
        [0, 1, 0, 1], # m3 -> c01, c11
        [1, 0, 1, 0], # m4 -> c00, c10
        [-1, 1, 0, 0], # m5 -> c00, c01
        [0, 0, 0, 1], # m6 -> c11
        [1, 0, 0, 0]  # m7 -> c00
    ]
    
    # Note: strassen_W[q][c_local] is the coefficient of m_q in c_local.
    # c_local 0: c00
    # c_local 1: c01
    # c_local 2: c10
    # c_local 3: c11
    
    for g in range(4):
        for q in range(7):
            r = g * 7 + q
            # Fill U
            for a_idx in range(4):
                U[a_idx][r] = strassen_A[q][a_idx]
            
            # Fill V
            # Local B indices:
            # 0: b00 (global j=0, k=2g)
            # 1: b01 (global j=0, k=2g+1)
            # 2: b10 (global j=1, k=2g)
            # 3: b11 (global j=1, k=2g+1)
            for b_local_idx in range(4):
                coeff = strassen_B[q][b_local_idx]
                if coeff != 0:
                    # Map local index to global row in V
                    if b_local_idx == 0:
                        j, k = 0, 2*g
                    elif b_local_idx == 1:
                        j, k = 0, 2*g + 1
                    elif b_local_idx == 2:
                        j, k = 1, 2*g
                    else:
                        j, k = 1, 2*g + 1
                    v_row = j * p + k
                    V[v_row][r] = coeff
            
            # Fill W
            # Local C indices:
            # 0: c00 (global i=0, k=2g)
            # 1: c01 (global i=0, k=2g+1)
            # 2: c10 (global i=1, k=2g)
            # 3: c11 (global i=1, k=2g+1)
            for c_local_idx in range(4):
                coeff = strassen_W[q][c_local_idx]
                if coeff != 0:
                    if c_local_idx == 0:
                        i, k = 0, 2*g
                    elif c_local_idx == 1:
                        i, k = 0, 2*g + 1
                    elif c_local_idx == 2:
                        i, k = 1, 2*g
                    else:
                        i, k = 1, 2*g + 1
                    w_row = k * n + i
                    W[w_row][r] = coeff
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
