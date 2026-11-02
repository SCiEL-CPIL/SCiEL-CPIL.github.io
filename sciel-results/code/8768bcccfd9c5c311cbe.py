"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition.

Strategy: 7 independent 2x2 blocks, each using Strassen (rank 7) -> total rank 49.
This is the theoretical minimum for this direct-sum structure.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin within 150s wall-clock

    size_a = n * m      # 4
    size_b = m * p      # 28
    size_c = p * n      # 28

    rank = 49
    U = np.zeros((size_a, rank), dtype=np.int64)
    V = np.zeros((size_b, rank), dtype=np.int64)
    W = np.zeros((size_c, rank), dtype=np.int64)

    # Build the true multiplication tensor for verification
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U_mat, V_mat, W_mat, atol=1e-9):
        T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
        return np.allclose(T, T_true, atol=atol)

    # Strassen decomposition for each 2x2 block
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # For block q (q=0..6), columns k0=2q, k1=2q+1
    # B indices: B0k0=k0, B0k1=k1, B1k0=p+k0, B1k1=p+k1
    # W indices: C0k0=k0*n, C1k0=k0*n+1, C0k1=k1*n, C1k1=k1*n+1

    for q in range(7):
        k0 = 2 * q
        k1 = 2 * q + 1
        base = 7 * q

        # B flat indices
        v_B00 = k0          # B[0, k0]
        v_B01 = k1          # B[0, k1]
        v_B10 = p + k0      # B[1, k0]
        v_B11 = p + k1      # B[1, k1]

        # W flat indices (C indices: k*n + i)
        w_C00 = k0 * n      # C[0, k0]
        w_C01 = k1 * n      # C[0, k1]
        w_C10 = k0 * n + 1  # C[1, k0]
        w_C11 = k1 * n + 1  # C[1, k1]

        # M1 = (A00 + A11)(B00 + B11)
        c = base + 0
        U[0, c] += 1; U[3, c] += 1
        V[v_B00, c] += 1; V[v_B11, c] += 1
        W[w_C00, c] += 1; W[w_C11, c] += 1

        # M2 = (A10 + A11) * B00
        c = base + 1
        U[2, c] += 1; U[3, c] += 1
        V[v_B00, c] += 1
        W[w_C10, c] += 1; W[w_C11, c] -= 1

        # M3 = A00 * (B01 - B11)
        c = base + 2
        U[0, c] += 1
        V[v_B01, c] += 1; V[v_B11, c] -= 1
        W[w_C01, c] += 1; W[w_C11, c] += 1

        # M4 = A11 * (B10 - B00)
        c = base + 3
        U[3, c] += 1
        V[v_B10, c] += 1; V[v_B00, c] -= 1
        W[w_C00, c] += 1; W[w_C10, c] += 1

        # M5 = (A00 + A01) * B11
        c = base + 4
        U[0, c] += 1; U[1, c] += 1
        V[v_B11, c] += 1
        W[w_C00, c] -= 1; W[w_C01, c] += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = base + 5
        U[2, c] += 1; U[0, c] -= 1
        V[v_B00, c] += 1; V[v_B01, c] += 1
        W[w_C11, c] += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = base + 6
        U[1, c] += 1; U[3, c] -= 1
        V[v_B10, c] += 1; V[v_B11, c] += 1
        W[w_C00, c] += 1

    # Verify the decomposition
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 56
    rank_sb = n * m * p  # 56
    U_sb = np.zeros((size_a, rank_sb), dtype=np.int64)
    V_sb = np.zeros((size_b, rank_sb), dtype=np.int64)
    W_sb = np.zeros((size_c, rank_sb), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
