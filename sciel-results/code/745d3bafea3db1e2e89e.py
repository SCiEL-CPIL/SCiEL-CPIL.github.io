import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _get_u_rows(n, m, bx, by):
    n2, m2 = n // 2, m // 2
    i_start, i_end = bx * n2, (bx + 1) * n2
    j_start, j_end = by * m2, (by + 1) * m2
    rows = []
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            rows.append(i * m + j)
    return rows

def _get_v_rows(m, p, bx, by):
    m2, p2 = m // 2, p // 2
    j_start, j_end = bx * m2, (bx + 1) * m2
    k_start, k_end = by * p2, (by + 1) * p2
    rows = []
    for j in range(j_start, j_end):
        for k in range(k_start, k_end):
            rows.append(j * p + k)
    return rows

def _get_w_rows(n, p, bx, by):
    n2, p2 = n // 2, p // 2
    i_start, i_end = bx * n2, (bx + 1) * n2
    k_start, k_end = by * p2, (by + 1) * p2
    rows = []
    for k in range(k_start, k_end):
        for i in range(i_start, i_end):
            rows.append(k * n + i)
    return rows

def _construct_recursive(n, m, p):
    # Base cases where any dimension is 1
    if n == 1 or m == 1 or p == 1:
        R = 1
        U = np.ones((n * m, R))
        V = np.ones((m * p, R))
        W = np.ones((p * n, R))
        return U, V, W

    # Recursive case
    n2, m2, p2 = n // 2, m // 2, p // 2
    Us, Vs, Ws = _construct_recursive(n2, m2, p2)
    Rs = Us.shape[1]
    R = 7 * Rs

    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Strassen terms: (u_blocks, v_blocks, w_blocks)
    # Each block is (bx, by, sign)
    terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M2 = (A10 + A11)B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1)]),
        # M3 = A00(B01 - B11)
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1)]),
        # M4 = A11(B10 - B00)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(1, 0, 1)]),
        # M5 = (A00 + A10)B11
        ([(0, 0, 1), (1, 0, 1)], [(1, 1, 1)], [(0, 1, 1)]),
        # M6 = (A01 - A11)(B00 + B10)
        ([(0, 1, 1), (1, 1, -1)], [(0, 0, 1), (1, 0, 1)], [(1, 1, 1)]),
        # M7 = (A10 - A00)(B11 + B01)
        ([(1, 0, 1), (0, 0, -1)], [(1, 1, 1), (0, 1, 1)], [(0, 1, 1)]),
    ]

    for t_idx, (u_b, v_b, w_b) in enumerate(terms):
        col_start = t_idx * Rs
        col_end = col_start + Rs

        # Fill U
        for bx, by, sign in u_b:
            rows = _get_u_rows(n, m, bx, by)
            U[rows, col_start:col_end] += sign * Us

        # Fill V
        for bx, by, sign in v_b:
            rows = _get_v_rows(m, p, bx, by)
            V[rows, col_start:col_end] += sign * Vs

        # Fill W
        for bx, by, sign in w_b:
            rows = _get_w_rows(n, p, bx, by)
            W[rows, col_start:col_end] += sign * Ws

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _construct_recursive(n, m, p)
    
    # Convert to lists of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
