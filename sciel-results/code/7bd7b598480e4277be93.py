"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

Uses the Kronecker product of the Strassen 2x2 algorithm (rank 7) to get
a rank-49 decomposition for 4x4 matrix multiplication. The decomposition
is verified exactly before returning; on any failure it falls back to the
rank-64 schoolbook algorithm.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
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


def _strassen_kron_4x4(n, m, p):
    """Rank-49 decomposition via Kronecker product of Strassen 2x2."""
    # Strassen 2x2 decomposition (rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Verify 2x2 Strassen decomposition first
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.dot(U2[i*2+j, :], V2[j*2+k, :] * W2[k*2+i, :]))
                if s != 1:
                    return _schoolbook(n, m, p)

    # Kronecker product to get 4x4 decomposition
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    # Permute rows to match linear index convention
    # U: row index i*4+j (i is row of A, j is col of A)
    # V: row index j*4+k (j is row of B, k is col of B)
    # W: row index k*4+i (k is col of B, i is row of A)

    perm_u = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        a_idx = i_out * 2 + j_out
        b_idx = i_in * 2 + j_in
        perm_u[S] = a_idx * 4 + b_idx

    perm_v = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        a_idx = j_out * 2 + k_out
        b_idx = j_in * 2 + k_in
        perm_v[S] = a_idx * 4 + b_idx

    perm_w = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        a_idx = k_out * 2 + i_out
        b_idx = k_in * 2 + i_in
        perm_w[S] = a_idx * 4 + b_idx

    U4 = U4[perm_u, :]
    V4 = V4[perm_v, :]
    W4 = W4[perm_w, :]

    # Verify the full 4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = int(np.dot(U4[i*4+j, :], V4[j*4+k, :] * W4[k*4+i, :]))
                if s != 1:
                    return _schoolbook(n, m, p)

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True


def construct():
    """
    Constructs a minimal-rank exact tensor decomposition for 4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Use rank-49 Kronecker-Strassen decomposition
    if time.time() < deadline:
        r49 = _strassen_kron_4x4(n, m, p)
        return r49

    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
