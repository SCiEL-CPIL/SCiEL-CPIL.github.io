"""
Rewritten rank-32 tensor decomposition for 2x2 * 2x9 matrix multiplication.

Approach:
- Partition the 9 output columns into 4 pairs + 1 single column.
- For each pair of columns (k1, k2), use Strassen's rank-7 algorithm on the
  2x2 * 2x2 sub-problem (treating the two columns as a 2x2 block).
- For the single remaining column, use the rank-4 schoolbook algorithm.
- Total rank: 4*7 + 4 = 32.

Index conventions:
- U has shape (n*m, rank) = (4, rank), indexed by (i*m + j) where i in {0,1}, j in {0,1}
- V has shape (m*p, rank) = (18, rank), indexed by (j*p + k) where j in {0,1}, k in {0..8}
- W has shape (p*n, rank) = (18, rank), indexed by (k*n + i) where k in {0..8}, i in {0,1}
"""

def construct():
    """
    Construct a rank-32 tensor decomposition for 2x2 * 2x9 matrix multiplication.
    """
    n, m, p = 2, 2, 9
    rank = 32

    U = [[0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank for _ in range(p * n)]   # 18 x 32

    def strassen_block(k1, k2, start_col):
        """Add 7 Strassen products for columns k1, k2 starting at column start_col."""
        # V row indices for this block:
        v_00 = 0 * p + k1  # b0k1
        v_01 = 0 * p + k2  # b0k2
        v_10 = 1 * p + k1  # b1k1
        v_11 = 1 * p + k2  # b1k2

        # W row indices for this block:
        w_00 = k1 * n + 0  # c_{0,k1}
        w_01 = k1 * n + 1  # c_{1,k1}
        w_02 = k2 * n + 0  # c_{0,k2}
        w_03 = k2 * n + 1  # c_{1,k2}

        # U row indices:
        u_00 = 0  # a00
        u_01 = 1  # a01
        u_10 = 2  # a10
        u_11 = 3  # a11

        # M1 = (a00 + a11) * (b0k1 + b1k2)
        c = start_col + 0
        U[u_00][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        W[w_03][c] += 1

        # M2 = (a10 + a11) * b0k1
        c = start_col + 1
        U[u_10][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_01][c] += 1
        W[w_03][c] -= 1

        # M3 = a00 * (b0k2 - b1k2)
        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1
        V[v_11][c] -= 1
        W[w_02][c] += 1
        W[w_03][c] += 1

        # M4 = a11 * (b1k1 - b0k1)
        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1
        V[v_00][c] -= 1
        W[w_00][c] += 1
        W[w_01][c] += 1

        # M5 = (a00 + a01) * b1k2
        c = start_col + 4
        U[u_00][c] += 1
        U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] -= 1
        W[w_02][c] += 1

        # M6 = (a10 - a00) * (b0k1 + b0k2)
        c = start_col + 5
        U[u_10][c] += 1
        U[u_00][c] -= 1
        V[v_00][c] += 1
        V[v_01][c] += 1
        W[w_03][c] += 1

        # M7 = (a01 - a11) * (b1k1 + b1k2)
        c = start_col + 6
        U[u_01][c] += 1
        U[u_11][c] -= 1
        V[v_10][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1

        return start_col + 7

    def schoolbook_single(k, start_col):
        """Add 4 schoolbook products for a single column k starting at column start_col."""
        # V row indices:
        v_0 = 0 * p + k  # b0k
        v_1 = 1 * p + k  # b1k

        # W row indices:
        w_0 = k * n + 0  # c_{0,k}
        w_1 = k * n + 1  # c_{1,k}

        # U row indices:
        u_00 = 0  # a00
        u_01 = 1  # a01
        u_10 = 2  # a10
        u_11 = 3  # a11

        # N1 = a00 * b0k
        c = start_col + 0
        U[u_00][c] += 1
        V[v_0][c] += 1
        W[w_0][c] += 1

        # N2 = a01 * b1k
        c = start_col + 1
        U[u_01][c] += 1
        V[v_1][c] += 1
        W[w_0][c] += 1

        # N3 = a10 * b0k
        c = start_col + 2
        U[u_10][c] += 1
        V[v_0][c] += 1
        W[w_1][c] += 1

        # N4 = a11 * b1k
        c = start_col + 3
        U[u_11][c] += 1
        V[v_1][c] += 1
        W[w_1][c] += 1

        return start_col + 4

    # Process 4 pairs: (0,1), (2,3), (4,5), (6,7)
    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)

    # Process single column: 8
    col = schoolbook_single(8, col)

    return {"U": U, "V": V, "W": W}
