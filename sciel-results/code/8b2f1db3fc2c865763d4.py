"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]
import time

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-84 decomposition for 4x4 * 4x6
    n, m, p = DIMENSIONS
    rank = 84  # 12 block-pairs * 7 Strassen terms
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for a0 in range(0, n, 2):
        for b0 in range(0, m, 2):
            for c0 in range(0, p, 2):
                # 7 Strassen terms
                # m1 = (A00+A11)(B00+B11)
                for (ri, ci) in [(a0, b0), (a0+1, b0+1)]:
                    U[ri*m+ci][col] = 1
                for (rj, ck) in [(b0, c0), (b0+1, c0+1)]:
                    V[rj*p+ck][col] = 1
                for (rk, ii) in [(c0, a0), (c0+1, a0+1)]:
                    W[rk*n+ii][col] = 1
                col += 1

                # m2 = (A10+A11) B00
                for (ri, ci) in [(a0+1, b0), (a0+1, b0+1)]:
                    U[ri*m+ci][col] = 1
                V[b0*p+c0][col] = 1
                W[c0*n+(a0+1)][col] = 1
                W[(c0+1)*n+(a0+1)][col] = -1
                col += 1

                # m3 = A00 (B01-B11)
                U[a0*m+b0][col] = 1
                for (rj, ck) in [(b0, c0+1), (b0+1, c0+1)]:
                    V[rj*p+ck][col] = 1
                W[c0+1][col] = 1  # c0+1, a0  ->  W[(c0+1)*n + a0]
                W[(c0+1)*n+(a0+1)][col] = 1
                col += 1

                # m4 = A11 (B10-B00)
                U[(a0+1)*m+(b0+1)][col] = 1
                for (rj, ck) in [(b0+1, c0), (b0, c0)]:
                    V[rj*p+ck][col] = 1
                    if rj == b0:
                        V[rj*p+ck][col] = -1
                W[c0*n+a0][col] = 1
                W[c0*n+(a0+1)][col] = 1
                col += 1

                # m5 = (A00+A01) B11
                for (ri, ci) in [(a0, b0), (a0, b0+1)]:
                    U[ri*m+ci][col] = 1
                V[(b0+1)*p+(c0+1)][col] = 1
                W[c0*n+a0][col] = -1
                W[(c0+1)*n+a0][col] = 1
                col += 1

                # m6 = (A10-A00) (B00+B01)
                U[(a0+1)*m+b0][col] = 1
                U[a0*m+b0][col] = -1
                for (rj, ck) in [(b0, c0), (b0, c0+1)]:
                    V[rj*p+ck][col] = 1
                W[(c0+1)*n+(a0+1)][col] = 1
                col += 1

                # m7 = (A01-A11) (B10+B11)
                U[a0*m+(b0+1)][col] = 1
                U[(a0+1)*m+(b0+1)][col] = -1
                for (rj, ck) in [(b0+1, c0), (b0+1, c0+1)]:
                    V[rj*p+ck][col] = 1
                W[c0*n+a0][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
