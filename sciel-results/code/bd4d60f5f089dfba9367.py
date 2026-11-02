"""Matrix multiplication 2x2 by 2x8 using a rank-7 decomposition."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8

    # Strassen coefficients for 2x2x2 matrix multiplication
    # u_coeffs[r][i][j] where i,j in {0,1}
    u_coeffs = [
        [[1, 0], [0, 1]],   # r0: (a+d)(e+h) -> u: a=1, d=1
        [[0, 0], [1, 1]],   # r1: (c+d)e    -> u: c=1, d=1
        [[1, 0], [0, 0]],   # r2: a(f-h)    -> u: a=1
        [[0, 0], [0, 1]],   # r3: d(g-e)    -> u: d=1
        [[1, 1], [0, 0]],   # r4: (a+b)h    -> u: a=1, b=1
        [[-1, 0], [1, 0]],  # r5: (c-a)(e+f)-> u: c=1, a=-1
        [[0, 1], [0, -1]]   # r6: (b-d)(g+h)-> u: b=1, d=-1
    ]

    # v_coeffs[r][j][k] where j,k in {0,1}
    v_coeffs = [
        [[1, 0], [0, 1]],   # r0: (a+d)(e+h) -> v: e=1, h=1
        [[1, 0], [0, 0]],   # r1: (c+d)e    -> v: e=1
        [[0, 1], [0, -1]],  # r2: a(f-h)    -> v: f=1, h=-1
        [[-1, 0], [1, 0]],  # r3: d(g-e)    -> v: e=-1, g=1
        [[0, 0], [0, 1]],   # r4: (a+b)h    -> v: h=1
        [[1, 1], [0, 0]],   # r5: (c-a)(e+f)-> v: e=1, f=1
        [[0, 0], [1, 1]]    # r6: (b-d)(g+h)-> v: g=1, h=1
    ]

    # w_coeffs[r][i][k] where i,k in {0,1}
    w_coeffs = [
        [[1, 0], [0, 1]],   # r0: C00+=1, C11+=1
        [[0, 0], [1, -1]],  # r1: C10+=1, C11-=-1
        [[0, 1], [0, 1]],   # r2: C01+=1, C11+=1
        [[1, 0], [1, 0]],   # r3: C00+=1, C10+=1
        [[-1, 1], [0, 0]],  # r4: C00-=-1, C01+=1
        [[0, 0], [0, 1]],   # r5: C11+=1
        [[1, 0], [0, 0]]    # r6: C00+=1
    ]

    rank = 7
    U = [[0] * rank for _ in range(n * m)]  # 4 x 7
    V = [[0] * rank for _ in range(m * p)]  # 16 x 7
    W = [[0] * rank for _ in range(p * n)]  # 16 x 7

    num_blocks = p // 2  # 4 blocks of width 2

    for r in range(rank):
        # Fill U: U[i*m + j][r] = u_coeffs[r][i][j]
        for i in range(2):
            for j in range(2):
                val = u_coeffs[r][i][j]
                if val != 0:
                    U[i * m + j][r] = val

        # Fill V: V[j*p + k][r] where k = block*2 + k_local
        # v_coeffs[r][j][k_local] is the coefficient for B[j, k_local]
        for j in range(2):
            for k_local in range(2):
                val = v_coeffs[r][j][k_local]
                if val != 0:
                    for block in range(num_blocks):
                        k_global = block * 2 + k_local
                        V[j * p + k_global][r] = val

        # Fill W: W[k*n + i][r] where k = block*2 + k_local
        # w_coeffs[r][i][k_local] is the coefficient for C[i, k_local]
        for i in range(2):
            for k_local in range(2):
                val = w_coeffs[r][i][k_local]
                if val != 0:
                    for block in range(num_blocks):
                        k_global = block * 2 + k_local
                        W[k_global * n + i][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
