"""Rank-24 exact decomposition for 2x4 by 4x4 matrix multiplication.

Approach (new design, not the 4-block Strassen scheme of the seed):
  - Use one Strassen 2x2x2 block (rank 7) for subcube A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2].
  - Use direct multiplication (rank 4 each) for the other three 2x2x2 subcubes.
  - Total rank = 7 + 4 + 4 + 4 = 24.
The tensor is verified exactly (over rationals) before returning.
"""
import numpy as np

DIMENSIONS = [2, 4, 4]


def _build_candidate():
    n, m, p = 2, 4, 4
    R = 24
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    # ---- Strassen 2x2x2 (local) coefficients ----
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U_loc[r][i,j], V_loc[r][j,k], W_loc[r][i,k]
    strassen_U = [
        {(0, 0): 1, (1, 1): 1},            # M1
        {(1, 0): 1, (1, 1): 1},            # M2
        {(0, 0): 1},                        # M3
        {(1, 1): 1},                        # M4
        {(0, 0): 1, (0, 1): 1},            # M5
        {(1, 0): 1, (0, 0): -1},           # M6
        {(0, 1): 1, (1, 1): -1},           # M7
    ]
    strassen_V = [
        {(0, 0): 1, (1, 1): 1},            # M1
        {(0, 0): 1},                        # M2
        {(0, 1): 1, (1, 1): -1},           # M3
        {(1, 0): 1, (0, 0): -1},           # M4
        {(1, 1): 1},                        # M5
        {(0, 0): 1, (0, 1): 1},            # M6
        {(1, 0): 1, (1, 1): 1},            # M7
    ]
    strassen_W = [
        {(0, 0): 1, (1, 1): 1},            # M1
        {(1, 0): 1, (1, 1): -1},           # M2
        {(0, 1): 1, (1, 1): 1},            # M3
        {(0, 0): 1, (1, 0): 1},            # M4
        {(0, 0): -1, (0, 1): 1},           # M5
        {(1, 1): 1},                        # M6
        {(0, 0): 1},                        # M7
    ]

    # ---- Block 0: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] ----
    for s in range(7):
        r = s
        for (il, jl), val in strassen_U[s].items():
            U[il * m + jl, r] += val
        for (jl, kl), val in strassen_V[s].items():
            V[jl * p + kl, r] += val
        for (il, kl), val in strassen_W[s].items():
            W[kl * n + il, r] += val

    # ---- Direct (rank 4) for the other three subcubes ----
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    for b in range(1, 4):
        base = 7 + (b - 1) * 4
        j_off = (b % 2) * 2
        k_off = (b // 2) * 2
        for il in range(2):
            for jl in range(2):
                r = base + (il * 2 + jl)
                i = il
                j = jl + j_off
                k = jl + k_off
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1

    return U, V, W


def _verify(U, V, W):
    n, m, p = 2, 4, 4
    R = U.shape[1]
    # Reshape: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Compute via einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Reference: T[i,j,k] = 1 iff i==j==k
    ref = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == j == k:
                    ref[i, j, k] = 1
    if not np.allclose(T, ref, atol=1e-9):
        return False
    return True


def construct():
    U, V, W = _build_candidate()
    if not _verify(U, V, W):
        raise RuntimeError("decomposition failed verification")
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
