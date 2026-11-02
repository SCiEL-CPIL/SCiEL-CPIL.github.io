"""Matrix multiplication 2,4,7 rank-minimal tensor decomposition.

Block decomposition: A (2x4) split into A_L (2x2) and A_R (2x2);
B (4x7) split into B_top (2x7) and B_bot (2x7).
- A_L x B_top uses Strassen (rank 7).
- A_R x B_bot uses schoolbook (rank 2*2*7 = 28).
Total rank = 7 + 28 = 35.
"""
DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    rank = 7 + n * n * p  # 7 + 2*2*7 = 35
    U = [[0] * rank for _ in range(n * m)]   # (8, rank)
    V = [[0] * rank for _ in range(m * p)]   # (28, rank)
    W = [[0] * rank for _ in range(p * n)]   # (14, rank)

    col = 0

    def add_strassen(a_terms, b_terms, c_terms):
        """a_terms: list of (i, j, sign); b_terms: list of (j, k, sign); c_terms: list of (i, k, sign)."""
        nonlocal col
        for (i, j, sa) in a_terms:
            for (jj, k, sb) in b_terms:
                U[i * m + j][col] = sa
                V[jj * p + k][col] = sb
        for (i, k, sc) in c_terms:
            W[k * n + i][col] = sc
        col += 1

    # Strassen on A_L (cols 0,1) x B_top (rows 0,1)
    add_strassen([(0, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (1, 1, 1)])
    add_strassen([(1, 0, 1), (1, 1, 1)],
                 [(0, 0, 1)],
                 [(1, 0, 1)])
    add_strassen([(0, 0, 1)],
                 [(0, 1, 1), (1, 1, -1)],
                 [(0, 1, 1), (1, 1, 1)])
    add_strassen([(1, 1, 1)],
                 [(1, 0, 1), (0, 0, -1)],
                 [(0, 0, -1), (1, 0, 1)])
    add_strassen([(0, 0, 1), (0, 1, 1)],
                 [(1, 1, 1)],
                 [(0, 0, -1), (1, 1, 1)])
    add_strassen([(1, 0, 1), (0, 0, -1)],
                 [(0, 0, 1), (0, 1, 1)],
                 [(1, 1, 1)])
    add_strassen([(0, 1, 1), (1, 1, -1)],
                 [(1, 0, 1), (1, 1, 1)],
                 [(0, 0, 1)])

    # Schoolbook on A_R (cols 2,3) x B_bot (rows 2,3)
    for i in range(n):
        for jj in range(n, m):
            for k in range(n, p):
                U[i * m + jj][col] = 1
                V[jj * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, (col, rank)
    return {"U": U, "V": V, "W": W}
