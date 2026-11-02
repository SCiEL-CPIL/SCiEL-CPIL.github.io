import time
import numpy as np

_DEADLINE = 115.0


def _strassen_triples():
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]


def _build_rank42():
    n, m, p = 2, 2, 12
    nblocks = p // 2
    rank = 7 * nblocks

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    triples = _strassen_triples()
    col = 0
    for b in range(nblocks):
        for (u, v, w) in triples:
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = u[i * 2 + j]
            for j in range(2):
                for k_loc in range(2):
                    V[j * p + (2 * b + k_loc), col] = v[j * 2 + k_loc]
            for k_loc in range(2):
                for i in range(2):
                    W[(2 * b + k_loc) * n + i, col] = w[k_loc * 2 + i]
            col += 1

    return U, V, W


def construct():
    t0 = time.time()
    U, V, W = _build_rank42()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
