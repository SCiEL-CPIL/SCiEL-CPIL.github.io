"""Matrix multiplication tensor decomposition for (2,3,10).

Strategy:
  * Split middle dimension m=3 into 2+1.
  * For each pair of output columns (k0, k0+1), apply Strassen rank-7
    to the 2x2x2 sub-problem (A[:,0:2] * B[0:2, k0:k1+1]).
  * Add schoolbook rank-4 for A[:,2] * B[2, k0:k1+1].
  * Total: 5 pairs * (7 + 4) = 55.
"""
import time
import random


def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55

    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # Strassen 7 products for X = A[:,0:2], Y = B[0:2, k0:k1+1]
        # X = [[a00, a01], [a10, a11]]
        # Y = [[B0k0, B0k1], [B1k0, B1k1]]
        # Output: C00 = C[0,k0], C01 = C[0,k1], C10 = C[1,k0], C11 = C[1,k1]

        # M1 = (a00 + a11) * (B0k0 + B1k1)
        U[0][col] += 1   # a00
        U[4][col] += 1   # a11
        V[k0][col] += 1       # B[0,k0]
        V[10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1

        # M2 = (a10 + a11) * B0k0
        U[3][col] += 1   # a10
        U[4][col] += 1   # a11
        V[k0][col] += 1       # B[0,k0]
        W[k0 * 2 + 1][col] += 1  # C[1,k0]
        W[k1 * 2 + 1][col] -= 1  # -C[1,k1]
        col += 1

        # M3 = a00 * (B1k0 - B0k0)
        U[0][col] += 1   # a00
        V[10 + k0][col] += 1  # B[1,k0]
        V[k0][col] -= 1       # -B[0,k0]
        W[k1 * 2 + 0][col] += 1  # C[0,k1]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        col += 1

        # M4 = a11 * (B1k0 - B0k0)
        U[4][col] += 1   # a11
        V[10 + k0][col] += 1  # B[1,k0]
        V[k0][col] -= 1       # -B[0,k0]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        W[k0 * 2 + 1][col] += 1  # C[1,k0]
        col += 1

        # M5 = (a00 + a01) * B1k0
        U[0][col] += 1   # a00
        U[1][col] += 1   # a01
        V[10 + k0][col] += 1  # B[1,k0]
        W[k0 * 2 + 0][col] -= 1  # -C[0,k0]
        W[k1 * 2 + 0][col] += 1  # C[0,k1]
        col += 1

        # M6 = (a10 - a00) * (B0k0 + B0k1)
        U[3][col] += 1   # a10
        U[0][col] -= 1   # -a00
        V[k0][col] += 1       # B[0,k0]
        V[k1][col] += 1       # B[0,k1]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1

        # M7 = (a01 - a11) * (B1k0 + B1k1)
        U[1][col] += 1   # a01
        U[4][col] -= 1   # -a11
        V[10 + k0][col] += 1  # B[1,k0]
        V[10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        col += 1

        # Schoolbook for A[:,2] * B[2, k0:k1+1]
        # a02 * B[2,k0] -> C[0,k0]
        U[2][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # a02 * B[2,k1] -> C[0,k1]
        U[2][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # a12 * B[2,k0] -> C[1,k0]
        U[5][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # a12 * B[2,k1] -> C[1,k1]
        U[5][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

    # Verify with random inputs
    rng = random.Random(42)
    valid = True
    for trial in range(30):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C_direct = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                C_direct[i][k] = sum(A[i][j] * B[j][k] for j in range(m))
        C_decomp = [[0.0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(rank):
                    ua = sum(U[i * m + j][r] * A[i][j] for j in range(m))
                    vb = sum(V[j * p + k][r] * B[j][k] for j in range(m))
                    acc += W[k * n + i][r] * ua * vb
                C_decomp[i][k] = acc
        for i in range(n):
            for k in range(p):
                if abs(C_direct[i][k] - C_decomp[i][k]) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if time.time() > deadline - 5.0:
            break

    if not valid:
        # Fallback to schoolbook rank 60
        rank60 = n * m * p
        U60 = [[0] * rank60 for _ in range(n * m)]
        V60 = [[0] * rank60 for _ in range(m * p)]
        W60 = [[0] * rank60 for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U60[i * m + j][idx] = 1
                    V60[j * p + k][idx] = 1
                    W60[k * n + i][idx] = 1
                    idx += 1
        return {"U": U60, "V": V60, "W": W60}

    return {"U": U, "V": V, "W": W}
