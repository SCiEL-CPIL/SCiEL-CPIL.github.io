"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition: 4 independent 2x2x2 blocks, each of rank 7.
    # Total rank = 28 (down from 32 schoolbook).
    n, m, p = DIMENSIONS
    # B is 2x8 = 4 blocks of 2x2.
    num_blocks = p // 2  # 4
    rank = 7 * num_blocks  # 28

    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28

    # Strassen coefficients for 2x2 matrix multiplication:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # A indices: a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[2], d=A[1,1]=U[3]
    # For block q (0..3):
    #   B elements: e=B[0,4q]=V[4q], f=B[0,4q+1]=V[4q+1],
    #               g=B[1,4q]=V[8+4q], h=B[1,4q+1]=V[8+4q+1]
    #   C elements: C[0,4q]=W[8q], C[0,4q+1]=W[8q+1],
    #               C[1,4q]=W[8q+2], C[1,4q+1]=W[8q+3]

    for q in range(num_blocks):
        base = 7 * q
        e_idx = 4 * q
        f_idx = 4 * q + 1
        g_idx = 8 + 4 * q
        h_idx = 8 + 4 * q + 1
        c00_idx = 8 * q
        c01_idx = 8 * q + 1
        c10_idx = 8 * q + 2
        c11_idx = 8 * q + 3

        # M1 = (a+d)(e+h)
        col = base + 0
        U[0][col] += 1   # a
        U[3][col] += 1   # d
        V[e_idx][col] += 1  # e
        V[h_idx][col] += 1  # h
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1

        # M2 = (c+d)e
        col = base + 1
        U[2][col] += 1   # c
        U[3][col] += 1   # d
        V[e_idx][col] += 1  # e
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1

        # M3 = a(f-h)
        col = base + 2
        U[0][col] += 1   # a
        V[f_idx][col] += 1  # f
        V[h_idx][col] -= 1  # -h
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1

        # M4 = d(g-e)
        col = base + 3
        U[3][col] += 1   # d
        V[g_idx][col] += 1  # g
        V[e_idx][col] -= 1  # -e
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1

        # M5 = (a+b)h
        col = base + 4
        U[0][col] += 1   # a
        U[1][col] += 1   # b
        V[h_idx][col] += 1  # h
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1

        # M6 = (c-a)(e+f)
        col = base + 5
        U[2][col] += 1   # c
        U[0][col] -= 1   # -a
        V[e_idx][col] += 1  # e
        V[f_idx][col] += 1  # f
        W[c11_idx][col] += 1

        # M7 = (b-d)(g+h)
        col = base + 6
        U[1][col] += 1   # b
        U[3][col] -= 1   # -d
        V[g_idx][col] += 1  # g
        V[h_idx][col] += 1  # h
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
