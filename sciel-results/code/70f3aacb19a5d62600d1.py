"""Matrix multiplication 4,5,5 tensor decomposition using 2x2x2 Strassen blocks + schoolbook fringes."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Strassen's 7-term decomposition for 2x2x2
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    strassen_222 = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    def build_decomposition_222(blocks):
        covered = set()
        for (i0, j0, k0) in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0 + di, j0 + dj, k0 + dk))

        num_schoolbook = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        num_schoolbook += 1

        num_strassen = len(blocks) * 7
        rank = num_schoolbook + num_strassen

        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

        for (i0, j0, k0) in blocks:
            for (u_c, v_c, w_c) in strassen_222:
                a_idx = [(i0, j0), (i0, j0 + 1), (i0 + 1, j0), (i0 + 1, j0 + 1)]
                b_idx = [(j0, k0), (j0, k0 + 1), (j0 + 1, k0), (j0 + 1, k0 + 1)]
                c_idx = [(k0, i0), (k0, i0 + 1), (k0 + 1, i0), (k0 + 1, i0 + 1)]
                for la, cf in enumerate(u_c):
                    if cf != 0:
                        U[a_idx[la][0] * m + a_idx[la][1]][col] += cf
                for lb, cf in enumerate(v_c):
                    if cf != 0:
                        V[b_idx[lb][0] * p + b_idx[lb][1]][col] += cf
                for lc, cf in enumerate(w_c):
                    if cf != 0:
                        W[c_idx[lc][0] * n + c_idx[lc][1]][col] += cf
                col += 1

        assert col == rank, f"Column mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if s != 1:
                        return False
        return True

    # Optimal 8 blocks for 4x5x5
    blocks_optimal = [(0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2)]
    
    best_result = None
    best_rank = n * m * p + 1

    try:
        candidate = build_decomposition_222(blocks_optimal)
        if verify(candidate["U"], candidate["V"], candidate["W"]):
            best_result = candidate
            best_rank = len(candidate["U"][0])
    except Exception:
        pass

    # Fallback to schoolbook
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}

    if time.time() > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
