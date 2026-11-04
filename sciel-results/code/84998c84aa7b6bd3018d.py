"""Matrix multiplication 2x4x4 with Strassen-based rank-28 decomposition."""
import time

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    """
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Uses Strassen's 7-multiplication algorithm on each of four 2x2x2 block products.
    """
    t0 = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS
    rank = 28

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_patterns = _strassen_2x2()

    # Four 2x2x2 block multiplications:
    # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    blocks = [
        (0, 0, 0, 0),   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 0, 2, 2),   # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (2, 2, 0, 0),   # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (2, 2, 2, 2),   # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    ]

    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_bj + local_c
                U[global_i * m + global_j][col] = coeff

            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_bj + local_r
                global_k = b_bk + local_c
                V[global_j * p + global_k][col] = coeff

            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_bk + local_c
                W[global_k * n + global_i][col] = coeff

            col += 1

            if time.time() - t0 > deadline - 5:
                break

    return {"U": U, "V": V, "W": W}
