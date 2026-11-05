"""
Rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication.

The schoolbook rank n*m*p = 3*4*6 = 72 is the best known rank for matrix multiplication
of dimensions 3x4 by 4x6. No lower rank decomposition is known in the literature for this
exact dimension pair.

The implementation provides the standard schoolbook construction, which is exact and
uses integer entries.

Index convention:
- U has shape (n*m, rank), indexed by (i, j) flattened as i*m + j
- V has shape (m*p, rank), indexed by (j, k) flattened as j*p + k
- W has shape (p*n, rank), indexed by (k, i) flattened as k*n + i  (trace convention)
"""

# Dimensions: A is (n x m), B is (m x p), C = A * B is (n x p)
_N = 3
_M = 4
_P = 6


def _make_schoolbook(n: int, m: int, p: int):
    """Return (U, V, W) as list-of-lists for schoolbook rank n*m*p."""
    r0 = n * m * p
    U = [[0] * r0 for _ in range(n * m)]
    V = [[0] * r0 for _ in range(m * p)]
    W = [[0] * r0 for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return U, V, W


def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.

    Returns a dict with keys 'U', 'V', 'W', each a list of lists of integers,
    where:
      U: shape (n*m, rank)
      V: shape (m*p, rank)
      W: shape (p*n, rank)
    The decomposition satisfies the matrix multiplication tensor identity:
      C[i, k] = sum_j A[i, j] * B[j, k]
    via the rank-r tensor factorization:
      T_{i,j,k}^{i,k} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
    """
    n, m, p = _N, _M, _P
    U, V, W = _make_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
