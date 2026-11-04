"""4x4x6 matrix multiplication via Kronecker (Strassen 2x2 x I2) block decomposition."""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100.0  # safety margin before 120s kill

    n, m, p = DIMENSIONS

    # Rank-7 Strassen 2x2x2 tensor (a,b,c in {0,1}), T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    S = [
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ]
    U2 = [[0]*7 for _ in range(2)]
    V2 = [[0]*7 for _ in range(2)]
    W2 = [[0]*7 for _ in range(2)]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for r in range(7):
                    U2[a][r] += S[a][b][c][r]
                    V2[b][r] += S[a][b][c][r]
                    W2[c][r] += S[a][b][c][r]

    # Assemble 4x4x6 as 3 independent 4x4x2 blocks, each = Strassen(2x2) x I(2x2).
    rank = 3 * 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for block in range(3):          # block b: columns j in [2b, 2b+1], rows k in [2b, 2b+1]
        base = block * 7
        for i2 in range(2):         # 2x2 row block of A (rows 2*i2, 2*i2+1)
            for j2 in range(2):     # 2x2 col block of A (cols 2*j2, 2*j2+1)
                for r in range(7):
                    U[2*i2 + j2][base + r] = U2[i2][r]
                    V[2*j2 + 2*block + 0][base + r] = V2[j2][r]
                    V[2*j2 + 2*block + 1][base + r] = V2[j2][r]
                    for k2 in range(2):  # identity over 2x2 row block of C
                        W[(2*block + k2)*n + 2*i2 + k2][base + r] = W2[k2][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
