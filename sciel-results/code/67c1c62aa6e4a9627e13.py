"""Strassen-based rank-35 decomposition for 2x2 * 2x10 matrix multiplication."""
DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = 2, 2, 10
    rank = 35  # 5 blocks * 7 Strassen multiplications
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # Strassen coefficients for 2x2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # u_s: coefficients for A (a,b,c,d)
    # v_s: coefficients for B (e,f,g,h)
    # w_s: coefficients for C (c00, c01, c10, c11)
    U_coeffs = [
        [1, 0, 0, 1],  # M1: (a+d)
        [0, 0, 1, 1],  # M2: (c+d)
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: (a+b)
        [-1, 0, 1, 0], # M6: (c-a)
        [0, 1, 0, -1], # M7: (b-d)
    ]
    V_coeffs = [
        [1, 1, 1, 1],  # M1: (e+h)
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: (f-h)
        [0, -1, 1, 0], # M4: (g-f)
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: (e+f)
        [0, 0, 1, 1],  # M7: (g+h)
    ]
    W_coeffs = [
        [1, 0, 0, 1],  # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1], # M2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],  # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0],  # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0], # M5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],  # M6: C11 += 1
        [1, 0, 0, 0],  # M7: C00 += 1
    ]

    num_blocks = p // 2  # 5 blocks
    for t in range(num_blocks):
        for s in range(7):
            col = 7 * t + s
            # Set U column (same for all blocks)
            for i in range(4):
                U[i][col] = U_coeffs[s][i]
            # Set V column: B has m=2 rows, p=10 cols
            # Block t uses columns 2t and 2t+1 of B
            # V index = j*p + k, where j is row of B (0 or 1), k is col (0..9)
            # e = B[0][2t], f = B[0][2t+1], g = B[1][2t], h = B[1][2t+1]
            V[0 * 10 + (2 * t) + 0][col] = V_coeffs[s][0]   # e
            V[0 * 10 + (2 * t) + 1][col] = V_coeffs[s][1]   # f
            V[1 * 10 + (2 * t) + 0][col] = V_coeffs[s][2]   # g
            V[1 * 10 + (2 * t) + 1][col] = V_coeffs[s][3]   # h
            # Set W column: C has p=10 rows, n=2 cols
            # W index = k*n + i, where k is row of C (0..9), i is col (0 or 1)
            # C block t: rows 2t and 2t+1
            # c00 = C[2t][0], c01 = C[2t][1], c10 = C[2t+1][0], c11 = C[2t+1][1]
            W[(2 * t + 0) * 2 + 0][col] = W_coeffs[s][0]   # c00
            W[(2 * t + 0) * 2 + 1][col] = W_coeffs[s][1]   # c01
            W[(2 * t + 1) * 2 + 0][col] = W_coeffs[s][2]   # c10
            W[(2 * t + 1) * 2 + 1][col] = W_coeffs[s][3]   # c11

    return {"U": U, "V": V, "W": W}
