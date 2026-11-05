"""Matrix multiplication 3,3,7 — constructive rank-minimization.

Build a compact decomposition and eliminate linearly dependent terms.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]
_TOL = 1e-9
_DEADLINE = 120.0


def _matmul_tensor(n, m, p):
    return np.ones((n, m, p), dtype=float)


def _schoolbook(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _decompose_fast(T, U, V, W):
    """Fast elimination using vectorized tensor construction."""
    n, m, p = T.shape
    r = U.shape[1]
    U2 = U.copy()
    V2 = V.copy()
    W2 = W.copy()
    kept = []

    # Precompute index arrays for vectorized tensor construction
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)

    for c in range(r):
        Tc = (U2[i_idx * m + j_idx, c]
              * V2[j_idx * p + k_idx, c]
              * W2[k_idx * n + i_idx, c])
        flat = Tc

        if len(kept) == 0:
            kept.append(c)
            continue

        A = np.array([
            (U2[i_idx * m + j_idx, kc]
             * V2[j_idx * p + k_idx, kc]
             * W2[k_idx * n + i_idx, kc])
            for kc in kept
        ]).T

        A_aug = np.hstack([A, flat.reshape(-1, 1)])
        rA = np.linalg.matrix_rank(A, tol=_TOL)
        rAug = np.linalg.matrix_rank(A_aug, tol=_TOL)

        if rAug > rA:
            kept.append(c)
            coeffs, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
            for c2 in range(c + 1, r):
                U2[:, c2] -= np.dot(coeffs, U2[:, kept])
                V2[:, c2] -= np.dot(coeffs, V2[:, kept])
                W2[:, c2] -= np.dot(coeffs, W2[:, kept])
        else:
            U2[:, c] = 0
            V2[:, c] = 0
            W2[:, c] = 0

    Uf = U2[:, kept]
    Vf = V2[:, kept]
    Wf = W2[:, kept]

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(kept)):
                    s += Uf[i * m + j, c] * Vf[j * p + k, c] * Wf[k * n + i, c]
                if abs(s - T[i, j, k]) > _TOL:
                    raise ValueError("Decomposition failed")

    return Uf.tolist(), Vf.tolist(), Wf.tolist()


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    T = _matmul_tensor(n, m, p)

    # Start with schoolbook rank 63
    U, V, W = _schoolbook(n, m, p)

    Uf, Vf, Wf = _decompose_fast(T, U, V, W)

    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": Uf, "V": Vf, "W": Wf}
