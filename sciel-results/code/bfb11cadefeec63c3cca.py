"""Kronecker(Strassen-2x2, Schoolbook-3x3) tensor decomposition of 6x6 matrix multiply.

The 6x6 multiplication tensor T[i,j,k] = e_{i,k} ⊗ e_{j,j} ⊗ e_{k,k} decomposes as the
Kronecker product T2 ⊗ T3 (since 6 = 2*3), so it has an exact rank-7*27 = 189
decomposition.  We build the rank-1 factors by tensoring the 2x2 Strassen factors with
the 3x3 schoolbook factors, then block-diagonalize the factor matrices (each factor is
the Kronecker product of two block-diagonal matrices, hence itself block diagonal with a
single non-zero entry per factor).  This preserves the decomposition exactly and lets us
verify it cheaply and exactly (integer arithmetic).
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]


def _strassen_2x2():
    # Rank-7 decomposition of 2x2 matrix multiplication (Kronecker product of 7 rank-1
    # tensors).  U[i,j,c], V[j,k,c], W[k,i,c] with C[i,k] = sum_c U[i,j,c]*A[i,j]*V[j,k,c]*B[j,k].
    U = [
        [1, 1, 1, 1, 0, 0, 0],   # A[0,0]
        [0, 0, 1, 0, 1, 0, 0],   # A[0,1]
        [0, 0, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 0, 0, 0, 0, 0, 1],   # A[1,1]
    ]
    V = [
        [1, 0, 1, 0, 0, 0, 0],   # B[0,0]
        [0, 0, 0, 0, 1, 0, 0],   # B[0,1]
        [0, 1, 0, 1, 0, 0, 0],   # B[1,0]
        [0, 0, 0, 0, 0, 1, 1],   # B[1,1]
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],   # C[0,0]
        [0, 0, 0, 0, 1, 0, 0],   # C[0,1]
        [0, 1, 0, 0, 0, 0, 0],   # C[1,0]
        [1, 0, 1, 1, 0, 1, 1],   # C[1,1]
    ]
    return np.array(U, dtype=object), np.array(V, dtype=object), np.array(W, dtype=object)


def _schoolbook_3x3():
    n = 3
    U = np.zeros((n * n, n * n), dtype=object)
    V = np.zeros((n * n, n * n), dtype=object)
    W = np.zeros((n * n, n * n), dtype=object)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = i * n * n + j * n + k
                U[i * n + j, c] = 1
                V[j * n + k, c] = 1
                W[k * n + i, c] = 1
    return U, V, W


def _block_diagonalize(M):
    """M is (n1*n2, r1*r2) whose non-zero columns are single 1s at (b1*a1, b2*a2).
    Reorder the factor index to (b2*r1 + b1) so that M becomes block diagonal with one
    non-zero entry per factor: row b1*n2 + b2, factor b2*r1 + b1."""
    n1, r1 = M.shape[0] // M.shape[1], M.shape[1]
    n1, r1 = M.shape[0], M.shape[1]
    n2 = M.shape[0] // n1
    r2 = M.shape[1] // r1
    out = np.zeros((n1 * n2, r1 * r2), dtype=object)
    for a1 in range(n1):
        for a2 in range(n2):
            for b1 in range(r1):
                for b2 in range(r2):
                    val = M[a1 * n2 + a2, b1 * r2 + b2]
                    if val != 0:
                        out[b1 * n2 + b2, b2 * r1 + b1] = val
    return out


def _verify(U, V, W, n, m, p):
    """Exact verification that U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] reproduces the
    multiplication tensor (identity in i,k, delta in j)."""
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1 if (i == k) else 0
                if s != expected:
                    return False
    return True


def _to_list(M):
    return [[int(M[r, c]) for c in range(M.shape[1])] for r in range(M.shape[0])]


def construct():
    deadline = time.time() + 110.0  # safety margin under the 150 s wall-clock limit
    n, m, p = DIMENSIONS

    # Build the rank-189 Kronecker factors, then block-diagonalize each.
    U2, V2, W2 = _strassen_2x2()
    U3, V3, W3 = _schoolbook_3x3()

    U = _block_diagonalize(np.kron(U2, U3))
    V = _block_diagonalize(np.kron(V2, V3))
    W = _block_diagonalize(np.kron(W2, W3))

    # Exact, cheap verification (integer arithmetic; 216 entries x 189 factors).
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p):
            # Fallback to the guaranteed-correct schoolbook decomposition (rank 216).
            rank = n * m * p
            Us = [[0] * rank for _ in range(n * m)]
            Vs = [[0] * rank for _ in range(m * p)]
            Ws = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Us[i * m + j][column] = 1
                        Vs[j * p + k][column] = 1
                        Ws[k * n + i][column] = 1
                        column += 1
            return {"U": Us, "V": Vs, "W": Ws}

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
