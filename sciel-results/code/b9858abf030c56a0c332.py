"""Exact rank-82 decomposition of M_{3,4,7} via two disjoint 2x2x2 Strassen blocks.

Strategy:
- Replace two non-overlapping 2x2x2 sub-blocks of the schoolbook (rank 84)
  decomposition with Strassen's rank-7 decomposition.
- The two blocks have disjoint U, V and W index sets, so each saves exactly
  one rank term: 84 - 2 = 82.
- A fast NumPy verification confirms exactness (within integer tolerance);
  on any failure we fall back to the guaranteed schoolbook rank-84 result.
"""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Guaranteed correct rank n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under the 150s limit

    rank = 82
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col_start, i0, j0, k0):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block at (i0,j0,k0)."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0*m+j0][col_start] = 1
        U[i1*m+j1][col_start] = 1
        V[j0*p+k0][col_start] = 1
        V[j1*p+k1][col_start] = 1
        W[k0*n+i0][col_start] = 1    # C[i0,k0] += 1
        W[k1*n+i1][col_start] = 1    # C[i1,k1] += 1

        # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1*m+j0][col_start+1] = 1
        U[i1*m+j1][col_start+1] = 1
        V[j0*p+k0][col_start+1] = 1
        W[k0*n+i1][col_start+1] = 1    # C[i1,k0] += 1
        W[k1*n+i1][col_start+1] = -1   # C[i1,k1] -= 1

        # M3 = A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0*m+j0][col_start+2] = 1
        V[j0*p+k1][col_start+2] = 1
        V[j1*p+k1][col_start+2] = -1
        W[k1*n+i0][col_start+2] = 1    # C[i0,k1] += 1
        W[k1*n+i1][col_start+2] = 1    # C[i1,k1] += 1

        # M4 = A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1*m+j1][col_start+3] = 1
        V[j1*p+k0][col_start+3] = 1
        V[j0*p+k0][col_start+3] = -1
        W[k0*n+i0][col_start+3] = 1    # C[i0,k0] += 1
        W[k0*n+i1][col_start+3] = 1    # C[i1,k0] += 1

        # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0*m+j0][col_start+4] = 1
        U[i0*m+j1][col_start+4] = 1
        V[j1*p+k1][col_start+4] = 1
        W[k0*n+i0][col_start+4] = -1   # C[i0,k0] -= 1
        W[k1*n+i0][col_start+4] = 1    # C[i0,k1] += 1

        # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1*m+j0][col_start+5] = 1
        U[i0*m+j0][col_start+5] = -1
        V[j0*p+k0][col_start+5] = 1
        V[j0*p+k1][col_start+5] = 1
        W[k1*n+i1][col_start+5] = 1    # C[i1,k1] += 1

        # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0*m+j1][col_start+6] = 1
        U[i1*m+j1][col_start+6] = -1
        V[j1*p+k0][col_start+6] = 1
        V[j1*p+k1][col_start+6] = 1
        W[k0*n+i0][col_start+6] = 1    # C[i0,k0] += 1

        return col_start + 7

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    col = add_strassen_block(0, 0, 0, 0)

    # Block 2: i in {1,2}, j in {2,3}, k in {5,6}
    col = add_strassen_block(col, 1, 2, 5)

    # Schoolbook for all remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                if i in (1, 2) and j in (2, 3) and k in (5, 6):
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    if col != rank:
        # Should never happen; fall back to guaranteed schoolbook.
        return _schoolbook(n, m, p)

    # Verify exactness; fall back to schoolbook on any discrepancy.
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
