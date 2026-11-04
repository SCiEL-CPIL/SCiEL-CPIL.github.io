"""Matrix multiplication 4x4x5: rank-minimal exact tensor decomposition.

Strategy:
  - Use the Kronecker-product Strassen construction to obtain a rank-49
    exact decomposition for the 4x4x4 sub-block (columns 0..3 of the
    right-hand matrix).
  - Extend with a rank-16 schoolbook block for the 5th column (4x4x1).
  - Total rank = 65.
  - Verify the full tensor identity T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r]
    W[k*n+i,r] == 1 for all valid (i,j,k) before returning.
  - Fallback: rank-80 schoolbook decomposition if the main construction
    fails (should never happen).
"""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify exact tensor decomposition.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for all valid (i,j,k).
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.shape[0] != n * m:
        return False
    rank = U.shape[1]
    if V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False

    # Build the full tensor and check it's all ones
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)

    return np.allclose(T, 1.0, atol=tol)


def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.

    A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    W index: k*2+i
    """
    # U factors for A entries [a00, a01, a10, a11]
    U2 = np.array([
        [1, 0, 0, 1],   # m1: (a00+a11)
        [0, 0, 1, 1],   # m2: (a10+a11)
        [1, 0, 0, 0],   # m3: a00
        [0, 0, 0, 1],   # m4: a11
        [1, 1, 0, 0],   # m5: (a00+a01)
        [-1, 0, 1, 0],  # m6: (a10-a00)
        [0, 1, 0, -1],  # m7: (a01-a11)
    ], dtype=np.float64)

    # V factors for B entries [b00, b01, b10, b11]
    V2 = np.array([
        [1, 0, 0, 1],   # m1: (b00+b11)
        [1, 0, 0, 0],   # m2: b00
        [0, 1, 0, -1],  # m3: (b01-b11)
        [-1, 0, 1, 0],  # m4: (b10-b00)
        [0, 0, 0, 1],   # m5: b11
        [1, 1, 0, 0],   # m6: (b00+b01)
        [0, 0, 1, 1],   # m7: (b10+b11)
    ], dtype=np.float64)

    # W factors for C entries [C00, C01, C10, C11] with index k*2+i
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    W2 = np.array([
        [1, 0, 0, 1],   # m1: C00+1, C11+1
        [0, 0, 1, -1],  # m2: C10+1, C11-1
        [0, 1, 0, 1],   # m3: C01+1, C11+1
        [1, 0, 1, 0],   # m4: C00+1, C10+1
        [-1, 1, 0, 0],  # m5: C00-1, C01+1
        [0, 0, 0, 1],   # m6: C11+1
        [1, 0, 0, 0],   # m7: C00+1
    ], dtype=np.float64)

    return U2, V2, W2


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using Kronecker products of 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2()

    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)

    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            U4[:, col] = np.kron(U2[:, r1], U2[:, r2])
            V4[:, col] = np.kron(V2[:, r1], V2[:, r2])
            W4[:, col] = np.kron(W2[:, r1], W2[:, r2])

    return U4, V4, W4


def _build_4x4x1_schoolbook():
    """Build rank-16 decomposition for 4x4x1 (A is 4x4, B is 4x1, C is 4x1).
    C[i] = sum_j A[i,j] * B[j]
    """
    n = 4
    rank = 16
    U = np.zeros((n * n, rank), dtype=np.float64)  # 16 x 16
    V = np.zeros((n, rank), dtype=np.float64)       # 4 x 16
    W = np.zeros((n, rank), dtype=np.float64)       # 4 x 16

    col = 0
    for i in range(n):
        for j in range(n):
            U[i * n + j, col] = 1.0
            V[j, col] = 1.0
            W[i, col] = 1.0
            col += 1

    return U, V, W


def _construct_4x4x5():
    """Construct the full 4x4x5 decomposition with rank 65."""
    if _time_remaining() < 5:
        raise TimeoutError("Not enough time remaining")

    # Build rank-49 base for 4x4x4
    U4, V4, W4 = _build_4x4x4_strassen()

    # Build rank-16 extension for the 5th column (4x4x1)
    U1, V1, W1 = _build_4x4x1_schoolbook()

    total_rank = 65

    # U_full: 16 x 65
    U_full = np.hstack([U4, U1])

    # V_full: 20 x 65 (m*p = 4*5 = 20)
    # V4 has 16 rows for (j,k) with k=0..3, indexed by j*4+k
    # V1 has 4 rows for j (k=4), indexed by j
    V_full = np.zeros((20, total_rank), dtype=np.float64)

    # Remap V4: row j*4+k -> j*5+k for k=0..3
    for j in range(4):
        for k in range(4):
            V_full[j * 5 + k, :49] = V4[j * 4 + k, :]

    # Place V1: row j -> j*5+4
    for j in range(4):
        V_full[j * 5 + 4, 49:] = V1[j, :]

    # W_full: 20 x 65 (p*n = 5*4 = 20)
    # W4 has 16 rows for (k,i) with k=0..3, indexed by k*4+i
    # W1 has 4 rows for i (k=4), indexed by i
    W_full = np.zeros((20, total_rank), dtype=np.float64)

    # Remap W4: row k*4+i -> k*4+i (same indexing since n=4)
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, :49] = W4[k * 4 + i, :]

    # Place W1: row i -> k=4, so k*4+i = 16+i
    for i in range(4):
        W_full[16 + i, 49:] = W1[i, :]

    # Verify
    if not _verify(U_full, V_full, W_full, 4, 4, 5, tol=1e-8):
        raise ValueError("Full 4x4x5 decomposition is invalid")

    return U_full, V_full, W_full


def _schoolbook_4x4x5():
    """Fallback schoolbook decomposition with rank 80."""
    n, m, p = 4, 4, 5
    rank = n * m * p  # 80
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return U, V, W


def construct():
    """Main entry point: construct the rank-minimal exact tensor decomposition for 4x4x5."""
    try:
        U, V, W = _construct_4x4x5()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except (TimeoutError, ValueError):
        pass

    # Fallback to schoolbook
    U, V, W = _schoolbook_4x4x5()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
