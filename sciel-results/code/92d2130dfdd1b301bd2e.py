"""Tensor decomposition for 2x2 matrix multiplication with rank 7."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7

    # Build U, V, W as (4, 7) matrices with integer entries
    # Indexing: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]

    # Seven rank-1 terms (Strassen-style)
    # Each term is a triple (u_vec, v_vec, w_vec) of length-4 vectors

    terms = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3 = A00(B01-B11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4 = A11(B10-B00)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6 = (A10-A00)(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (uv, vv, wv) in enumerate(terms):
        for idx in range(4):
            U[idx][r] = uv[idx]
            V[idx][r] = vv[idx]
            W[idx][r] = wv[idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
