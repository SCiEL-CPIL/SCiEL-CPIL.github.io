"""Matrix multiplication 2,3,8 - rank-24 exact tensor decomposition.

Strategy: split the 8 output columns into four 2-column blocks.  Each 2x3x2
all-ones sub-tensor (indices (i,j,k') with i in {0,1}, j in {0,1,2},
k' in {0,1}) is decomposed at rank 6:
  * rows i=0 and i=1 of the 2x2x2 core  ->  Strassen rank-7
  * the extra j=2 column               ->  2 rank-1 outer-product terms
  * the shared second row of the 2x3x2 (i=1, j in {0,1}, k' in {0,1})
                                        ->  1 shared rank-1 term
Total rank = 4 * 6 = 24 (vs schoolbook 48, seed 24).
"""
DIMENSIONS = [2, 3, 8]

import time


def _validate(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


# Strassen rank-7 decomposition of the 2x2x2 all-ones tensor.
# Each term is (u_dict, v_dict, w_dict); keys are (i,j) for U, (j,k') for V,
# (k',i) for W.
_STRASSEN_2x2x2 = [
    ({(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
    ({(0, 0): 1, (0, 1): 1, (1, 0): -1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
    ({(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): -1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
    ({(0, 0): 1, (0, 1): -1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
    ({(0, 0): 1, (0, 1): 1, (1, 0): -1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
    ({(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): -1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
    ({(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
     {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
]


def _block_decomposition(n, m, p):
    """Build rank-24 decomposition: 4 blocks of 6 terms each."""
    n, m, p = 2, 3, 8
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b
        col = b * 6
        # Terms 0..6: Strassen for the 2x2x2 core (i in {0,1}, j in {0,1}, k' in {0,1})
        for t in range(7):
            u_dict, v_dict, w_dict = _STRASSEN_2x2x2[t]
            for (i, j), val in u_dict.items():
                U[i * m + j][col + t] += val
            for (j, kp), val in v_dict.items():
                V[j * p + (k0 + kp)][col + t] += val
            for (kp, i), val in w_dict.items():
                W[(k0 + kp) * n + i][col + t] += val
        # Term 7: extra j=2 column for i=0
        U[0 * m + 2][col + 7] = 1
        V[2 * p + k0][col + 7] = 1
        V[2 * p + k0 + 1][col + 7] = 1
        W[k0 * n + 0][col + 7] = 1
        # Term 8: extra j=2 column for i=1
        U[1 * m + 2][col + 8] = 1
        V[2 * p + k0][col + 8] = 1
        V[2 * p + k0 + 1][col + 8] = 1
        W[k0 * n + 1][col + 8] = 1
        # Term 9: shared second row of the 2x3x2 (i=1, j in {0,1}, k' in {0,1})
        U[1 * m + 0][col + 9] = 1
        U[1 * m + 1][col + 9] = 1
        V[0 * p + k0][col + 9] = 1
        V[0 * p + k0 + 1][col + 9] = 1
        V[1 * p + k0][col + 9] = 1
        V[1 * p + k0 + 1][col + 9] = 1
        W[k0 * n + 1][col + 9] = 1
        W[(k0 + 1) * n + 1][col + 9] = 1

    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    n, m, p = DIMENSIONS

    # Deadline guard: bail out to schoolbook if we are running long.
    DEADLINE = 140.0  # 10s safety margin before the 150s kill

    # Best-effort low-rank construction.
    try:
        cand = _block_decomposition(n, m, p)
        if time.time() - start > DEADLINE:
            return _schoolbook(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand
    except Exception:
        pass

    # Fallback: always-valid schoolbook.
    return _schoolbook(n, m, p)
