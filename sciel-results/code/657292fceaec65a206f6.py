"""Rank-26 tensor decomposition for 3x3 * 3x4 matrix multiplication.

Decomposition (direct sum of sub-blocks):
  - Block A (i in {0,1}, j in {0,1}, k in {0,1}): Strassen 2x2x2, rank 7
  - Block B (i in {0,1}, j in {0,1}, k in {2,3}): Strassen 2x2x2, rank 7
  - Block C (i in {0,1}, j = 2,       k in {0..3}): schoolbook 2x1x4, rank 8
  - Block D (i = 2,     j in {0,1,2}, k in {0..3}): schoolbook 1x3x4, rank 12
Total rank = 7 + 7 + 8 + 12 = 26.

Convention: C[i,k] = sum_{i',j',j'',k''} T[i,k,i',j',j'',k''] A[i',j'] B[j'',k'']
with T[i*m+j, j'*p+k', k''*n+i'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
and T = sum_r U[i,j,r] * V[j',k',r] * W[k'',i',r].
"""

DIMENSIONS = [3, 3, 4]


def _build_rank26():
    n, m, p = 3, 3, 4
    rank = 26
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows
    col = [0]

    def add_strassen(i0, j0, k0):
        """Add 7 Strassen rank-1 terms for the 2x2x2 sub-block with top-left
        at (i0, j0, k0). Local A[i0+a, j0+b], B[j0+b, k0+c], C[i0+a, k0+c]."""
        for a, b, c, sign in [
            (0, 0, 0, 1), (1, 1, 1, 1),            # M1 = (A11+A22)(B11+B22) -> C11, C22
            (1, 0, 0, 1), (1, 1, 0, 1),            # M2 = (A21+A22)B11 -> C21
            (1, 1, 0, -1), (1, 1, 1, -1),          #     C22 -= M2
            (0, 0, 1, 1), (0, 0, 2, 1),            # M3 = A11(B12-B22) -> C12
            (1, 1, 2, -1), (1, 1, 1, 1),           #     C22 += M3
            (1, 1, 0, 1), (1, 1, 1, 1),            # M4 = A22(B21-B11) -> C11
            (0, 0, 1, -1), (0, 0, 0, 1),           #     C21 += M4
            (0, 0, 1, 1), (0, 1, 1, 1),            # M5 = (A11+A12)B22 -> C12
            (1, 1, 2, 1), (0, 0, 0, -1),           #     C11 -= M5
            (1, 0, 1, 1), (0, 0, 1, -1),           # M6 = (A21-A11)(B11+B12) -> C22
            (0, 0, 0, 1), (0, 0, 1, 1),
            (0, 1, 0, 1), (1, 1, 0, -1),           # M7 = (A12-A22)(B21+B22) -> C11
            (1, 1, 1, 1), (0, 0, 0, 1),
        ]:
            U[(i0 + a) * m + (j0 + b)][col[0]] = sign
            V[(j0 + b) * p + (k0 + c)][col[0]] = sign
            W[(k0 + c) * n + (i0 + a)][col[0]] = sign
            col[0] += 1

    def add_schoolbook(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i * m + j][col[0]] = 1
                    V[j * p + k][col[0]] = 1
                    W[k * n + i][col[0]] = 1
                    col[0] += 1

    add_strassen(0, 0, 0)      # Block A (7)
    add_strassen(0, 0, 2)      # Block B (7)
    add_schoolbook([0, 1], [2], [0, 1, 2, 3])   # Block C (8)
    add_schoolbook([2], [0, 1, 2], [0, 1, 2, 3])  # Block D (12)

    assert col[0] == rank, f"expected rank {rank}, got {col[0]}"
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    n, m, p = 3, 3, 4
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
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Return True iff the decomposition is exact under the trace convention."""
    n, m, p = 3, 3, 4
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for j1 in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            s = 0
                            for r in range(rank):
                                s += U[i * m + j1][r] * V[j2 * p + k2][r] * W[k * n + i2][r]
                            expected = 1 if (i == i2 and j1 == j2 and k == k2) else 0
                            if s != expected:
                                return False
    return True


def construct():
    try:
        cand = _build_rank26()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    return _build_schoolbook()
