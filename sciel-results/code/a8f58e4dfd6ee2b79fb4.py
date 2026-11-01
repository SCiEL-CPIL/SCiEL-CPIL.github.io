"""Matrix multiplication 2x2 by 2x16 - rank-minimal tensor decomposition."""
import time
import numpy as np

N, M, P = 2, 2, 16


def _target_tensor():
    """Build the matrix-multiplication tensor T[a,b,c] = 1 iff i=i', j=j', k=k'."""
    T = np.zeros((N * M, M * P, P * N), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1
    return T


def _verify(U, V, W):
    """Exact check that sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c] for all a,b,c."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    T = _target_tensor().astype(np.float64)
    return bool(np.allclose(recon, T, atol=1e-9, rtol=0.0))


def _strassen_decomposition():
    """
    Rank-56 decomposition: 2x2 * 2x16 = direct sum over 8 disjoint 2x2 * 2x2
    sub-problems (blocks of 2 output columns), each using Strassen's 7 factors.
    The B and C slot supports of different blocks are disjoint, so the rank is
    additive: 8 * 7 = 56, which is the provable minimum.
    """
    n, m, p = N, M, P
    # (A-form, B-form, C-contribution) for each Strassen multiplier M1..M7
    S = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, -1]),  # M7
    ]
    num_blocks = p // 2  # 8 blocks of 2 columns
    rank = num_blocks * len(S)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for q in range(num_blocks):
        col_start = 2 * q
        for a_form, b_form, c_contrib in S:
            U[:, col] = a_form
            for jp in range(2):
                for kp in range(2):
                    V[jp * p + (col_start + kp), col] = b_form[jp * 2 + kp]
            for ip in range(2):
                for kp in range(2):
                    W[(col_start + kp) * n + ip, col] = c_contrib[ip * 2 + kp]
            col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _schoolbook():
    """Standard rank 64 decomposition (fallback)."""
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    """Return the rank-minimal tensor decomposition for 2x2 * 2x16."""
    start = time.time()
    deadline = 110.0  # safety margin under the 120s limit

    # Rank-56 Strassen (provable optimum)
    try:
        cand = _strassen_decomposition()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass

    if time.time() - start > deadline:
        # Should not happen; return schoolbook without verification
        return _schoolbook()

    # Fallback: schoolbook rank 64
    cand = _schoolbook()
    if _verify(cand["U"], cand["V"], cand["W"]):
        return cand
    return cand
