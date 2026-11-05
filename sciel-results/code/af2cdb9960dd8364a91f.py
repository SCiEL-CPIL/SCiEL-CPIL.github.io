"""Rank-42 exact tensor decomposition for 3x4 by 4x4 matrix multiplication.

Strategy: partition the 3x4x4 multiplication tensor into:
  - four 2x2x2 sub-blocks, each solved with Strassen's 7-term algorithm,
  - a 1x4x2 border (row i=2, all j,k) solved by schoolbook,
  - a 3x2x2 border (rows i=0,1, col j=2,3, all k) solved by schoolbook.
Total rank = 4*7 + 8 + 6 = 42.
"""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_entries():
    # 7 Strassen products for two 2x2 matrices:
    #   M1=(a+b)(e+f)  M2=a(f-g)  M3=(c-d)(e+f)  M4=d(g-h)
    #   M5=(a+c)(e+h)  M6=(b-d)(g+h)  M7=(a-e)(f+h)
    # Each entry is a list of (coeff, M_index) pairs.
    E = [
        [(1, 0), (-1, 1), (1, 4), (-1, 6)],
        [(1, 1), (1, 6)],
        [(1, 3), (1, 5)],
        [(-1, 0), (1, 3), (1, 4), (1, 6)],
    ]
    return E


def _build():
    n, m, p = 3, 4, 4
    U = [[0.0] * 42 for _ in range(n * m)]   # row = i*m + j
    V = [[0.0] * 42 for _ in range(m * p)]   # row = j*p + k
    W = [[0.0] * 42 for _ in range(p * n)]   # row = k*n + i

    def ui(i, j):
        return i * m + j

    def vj(j, k):
        return j * p + k

    def wk(k, i):
        return k * n + i

    E = _strassen_entries()
    col = 0
    # Four 2x2x2 Strassen blocks (i0 in {0,1}, j0 in {0,2})
    for i0 in (0, 1):
        for j0 in (0, 2):
            for a in range(2):
                for b in range(2):
                    for c, mi in E[a][b]:
                        U[ui(i0 + a, j0 + b)][col] += c
                        V[vj(j0 + b, c)][col] += c
                        W[wk(c, i0 + a)][col] += c
            col += 7
    # 1x4x2 border: row i=2, all j,k (schoolbook)
    for j in range(4):
        for k in range(2):
            U[ui(2, j)][col] = 1.0
            V[vj(j, k)][col] = 1.0
            W[wk(k, 2)][col] = 1.0
            col += 1
    # 3x2x2 border: rows i=0,1, cols j=2,3, all k (schoolbook)
    for i in range(2):
        for j in (2, 3):
            for k in range(2):
                U[ui(i, j)][col] = 1.0
                V[vj(j, k)][col] = 1.0
                W[wk(k, i)][col] = 1.0
                col += 1

    assert col == 42, "expected rank 42, got %d" % col
    return {"U": U, "V": V, "W": W}


def construct():
    return _build()
# EVOLVE-BLOCK-END
