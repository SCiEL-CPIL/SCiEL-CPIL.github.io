"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # Decompose (2,4,8) as sum over g in {0,1} of (2,2,8) multiplications,
    # each (2,2,8) = 2x2 matrix multiplication over (2,1,4) sub-tensors,
    # decomposed by Strassen (rank 7) -> total rank 4(b) * 2(g) * 7 = 56.
    rank = 4 * 2 * 7  # 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # Strassen coefficients
    # alpha[t][i][h] for t in 0..6, i in 0..1, h in 0..1
    alpha = {
        0: {(0,0): 1, (1,1): 1},          # P1 = (M11+M22)(N11+N22)
        1: {(1,0): 1, (1,1): 1},          # P2 = (M21+M22)N11
        2: {(0,0): 1},                     # P3 = M11(N12-N22)
        3: {(1,1): 1},                     # P4 = M22(N21-N11)
        4: {(0,0): 1, (0,1): 1},          # P5 = (M11+M12)N22
        5: {(1,0): 1, (0,0): -1},         # P6 = (M21-M11)(N11+N12)
        6: {(0,1): 1, (1,1): -1},         # P7 = (M12-M22)(N21+N22)
    }
    beta = {
        0: {(0,0): 1, (1,1): 1},          # P1
        1: {(0,0): 1},                     # P2
        2: {(0,1): 1, (1,1): -1},         # P3
        3: {(1,0): 1, (0,0): -1},         # P4
        4: {(1,1): 1},                     # P5
        5: {(0,0): 1, (0,1): 1},          # P6
        6: {(1,0): 1, (1,1): 1},          # P7
    }
    # c[t][i][a]
    c = {
        0: {(0,0): 1, (1,1): 1},          # P1: D00+=P1, D11+=P1
        1: {(1,0): 1, (1,1): -1},         # P2: D10+=P2, D11-=P2
        2: {(0,1): 1, (1,1): 1},          # P3: D01+=P3, D11+=P3
        3: {(0,0): 1, (1,0): 1},          # P4: D00+=P4, D10+=P4
        4: {(0,0): -1, (0,1): 1},         # P5: D00-=P5, D01+=P5
        5: {(1,1): 1},                     # P6: D11+=P6
        6: {(0,0): 1},                     # P7: D00+=P7
    }

    col = 0
    for b in range(4):       # k = 4a + b, b in 0..3
        for g in range(2):   # j = 2g + h, g in 0..1
            for t in range(7):
                # alpha: U[i*m + j, col] = alpha[t][i][h], j = 2g+h
                for (i, h), val in alpha[t].items():
                    j = 2 * g + h
                    U[i * m + j][col] = val
                # beta: V[j*p + k, col] = beta[t][h][a], j = 2g+h, k = 4a+b
                for (h, a), val in beta[t].items():
                    j = 2 * g + h
                    k = 4 * a + b
                    V[j * p + k][col] = val
                # c: W[k*n + i, col] = c[t][i][a], k = 4a+b
                for (i, a), val in c[t].items():
                    k = 4 * a + b
                    W[k * n + i][col] = val
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
