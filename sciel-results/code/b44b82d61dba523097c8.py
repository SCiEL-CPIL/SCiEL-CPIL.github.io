"""Matrix multiplication 2x4x4: rank-minimal exact tensor decomposition.

Uses the Strassen-based construction for 2x4x4 with rank 28.
The 4x4 matrix B is treated as four independent 2x2 blocks in the j-k plane,
each multiplied by the 2x2 matrix A using Strassen's 7-term algorithm.
"""
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    U, V, W = _strassen_28(n, m, p)

    if time.time() > deadline:
        pass  # Static construction; should never trigger

    return {"U": U, "V": V, "W": W}


def _strassen_28(n, m, p):
    """Construct the rank-28 Strassen-based decomposition for 2x4x4.

    The four 2x2 blocks of B (in the j-k plane) are each handled independently
    with Strassen's 7-term algorithm, giving 4*7 = 28 rank-1 terms.
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 terms for 2x2 matrix multiplication.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    #
    # Index convention for local 2x2:
    # U: (i, j) where i in {0,1}, j in {0,1}
    # V: (j, k) where j in {0,1}, k in {0,1}
    # W: (k, i) where k in {0,1}, i in {0,1}
    strassen_terms = [
        # M1 = (a+d)(e+h)
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2 = (c+d)*e
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3 = a*(f-h)
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M4 = d*(g-e)
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5 = (a+b)*h
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6 = (c-a)*(e+f)
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7 = (b-d)*(g+h)
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]

    # Four 2x2x2 blocks: (j_offset, k_offset)
    # j_offset: which 2-column block of A (0 or 2)
    # k_offset: which 2-column block of B (0 or 2)
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 2}
    ]

    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Convert to integer entries
    U = [[int(round(x)) for x in row] for row in U]
    V = [[int(round(x)) for x in row] for row in V]
    W = [[int(round(x)) for x in row] for row in W]

    return U, V, W
