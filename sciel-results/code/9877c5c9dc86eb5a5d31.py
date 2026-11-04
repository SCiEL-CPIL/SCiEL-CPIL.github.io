"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Use a block structure: split each 6x6 matrix into 3x3 blocks of 2x2.
    # Apply Strassen (rank 7) to each 2x2 block multiplication within the 3x3 block grid.
    # Total rank = 3*3*3 * 7 = 189.
    rank = 189

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for i in range(3):
        for s in range(3):
            for j in range(3):
                for k in range(7):
                    # A block indices: rows 2i, 2i+1; cols 2s, 2s+1
                    a = (2 * i) * 6 + (2 * s)
                    b = (2 * i) * 6 + (2 * s + 1)
                    c = (2 * i + 1) * 6 + (2 * s)
                    d = (2 * i + 1) * 6 + (2 * s + 1)

                    # B block indices: rows 2s, 2s+1; cols 2j, 2j+1
                    e = (2 * s) * 6 + (2 * j)
                    f = (2 * s) * 6 + (2 * j + 1)
                    g = (2 * s + 1) * 6 + (2 * j)
                    h = (2 * s + 1) * 6 + (2 * j + 1)

                    # C block indices: W[k*n + r], so col*n + row
                    w00 = (2 * j) * 6 + (2 * i)
                    w01 = (2 * j + 1) * 6 + (2 * i)
                    w10 = (2 * j) * 6 + (2 * i + 1)
                    w11 = (2 * j + 1) * 6 + (2 * i + 1)

                    # Strassen products:
                    # M1 = (a+d)(e+h)
                    # M2 = (c+d)e
                    # M3 = a(f-h)
                    # M4 = d(g-e)
                    # M5 = (a+b)h
                    # M6 = (c-a)(e+f)
                    # M7 = (b-d)(g+h)

                    if k == 0:
                        U[a, col] = 1; U[d, col] = 1
                        V[e, col] = 1; V[h, col] = 1
                        W[w00, col] = 1; W[w11, col] = 1
                    elif k == 1:
                        U[c, col] = 1; U[d, col] = 1
                        V[e, col] = 1
                        W[w10, col] = 1; W[w11, col] = -1
                    elif k == 2:
                        U[a, col] = 1
                        V[f, col] = 1; V[h, col] = -1
                        W[w01, col] = 1; W[w11, col] = 1
                    elif k == 3:
                        U[d, col] = 1
                        V[g, col] = 1; V[e, col] = -1
                        W[w00, col] = 1; W[w10, col] = 1
                    elif k == 4:
                        U[a, col] = 1; U[b, col] = 1
                        V[h, col] = 1
                        W[w00, col] = -1; W[w01, col] = 1
                    elif k == 5:
                        U[c, col] = 1; U[a, col] = -1
                        V[e, col] = 1; V[f, col] = 1
                        W[w11, col] = 1
                    elif k == 6:
                        U[b, col] = 1; U[d, col] = -1
                        V[g, col] = 1; V[h, col] = 1
                        W[w00, col] = 1

                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
