"""Matrix multiplication 2x4x5 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 5

    # Each rank-1 term is (u_dict, v_dict, w_dict) with integer entries.
    # u keys: (i, j) for A[i][j]; v keys: (j, k) for B[j][k];
    # w keys: (k, i) for C[i][k] (W index = k*n + i).
    terms = []

    def term(u, v, w):
        terms.append((u, v, w))

    # Strassen 2x2x2 block: A rows i0,i1; B cols j0,j1; C cols k0,k1.
    def strassen_222(i0, i1, j0, j1, k0, k1):
        a0, a1 = (i0, j0), (i0, j1)
        a2, a3 = (i1, j0), (i1, j1)
        b0, b1 = (j0, k0), (j0, k1)
        b2, b3 = (j1, k0), (j1, k1)
        c00, c01 = (k0, i0), (k1, i0)
        c10, c11 = (k0, i1), (k1, i1)
        # M1=(a0+a3)(b0+b3) -> C00+1, C11+1
        term({a0: 1, a3: 1}, {b0: 1, b3: 1}, {c00: 1, c11: 1})
        # M2=(a2+a3)b0 -> C10+1, C11-1
        term({a2: 1, a3: 1}, {b0: 1}, {c10: 1, c11: -1})
        # M3=a1(b2+b3) -> C01+1, C11+1
        term({a1: 1}, {b2: 1, b3: 1}, {c01: 1, c11: 1})
        # M4=a0(b1-b2) -> C00+1, C10+1
        term({a0: 1}, {b1: 1, b2: -1}, {c00: 1, c10: 1})
        # M5=(a0+a1)b2 -> C01+1, C00-1
        term({a0: 1, a1: 1}, {b2: 1}, {c01: 1, c00: -1})
        # M6=(a3-a2)(b0+b1) -> C11+1
        term({a3: 1, a2: -1}, {b0: 1, b1: 1}, {c11: 1})
        # M7=(a1-a3)(b1+b3) -> C00+1
        term({a1: 1, a3: -1}, {b1: 1, b3: 1}, {c00: 1})

    # Schoolbook 2x2x1 block: A rows i0,i1; B cols j0,j1; C col k0.
    def schoolbook_221(i0, i1, j0, j1, k0):
        a0, a1 = (i0, j0), (i0, j1)
        a2, a3 = (i1, j0), (i1, j1)
        b0, b1 = (j0, k0), (j1, k0)
        c00, c10 = (k0, i0), (k0, i1)
        term({a0: 1}, {b0: 1}, {c00: 1})
        term({a1: 1}, {b1: 1}, {c00: 1})
        term({a2: 1}, {b0: 1}, {c10: 1})
        term({a3: 1}, {b1: 1}, {c10: 1})

    # Split m=4 into two m=2 blocks; each 2x2x5 split into p=[2,2,1].
    for bm in range(2):
        j0 = 2 * bm
        j1 = j0 + 1
        strassen_222(0, 1, j0, j1, 0, 1)
        strassen_222(0, 1, j0, j1, 2, 3)
        schoolbook_221(0, 1, j0, j1, 4)

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][c] += val
        for (j, k), val in v.items():
            V[j * p + k][c] += val
        for (k, i), val in w.items():
            W[k * n + i][c] += val
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
