"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Explicit rank-7 Strassen factors for 2x2x2 under the trace convention.

    T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{i,i'}.
    U row=(i,j), V row=(j',k), W row=(k',i').
    """
    U2 = np.array([
        [1, 1, 0, 0, 0, -1, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, -1],
    ], dtype=int)
    V2 = np.array([
        [0, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [-1, 1, -1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0],
    ], dtype=int)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=int)
    return U2, V2, W2


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Exact identity check: sum_c U[(i,j),c] V[(j',k),c] W[(k',i'),c]
       == delta_{j,j'} delta_{i,i'} for all i,j,j',k,k',i'."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    maxerr = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (jp == j and ip == i) else 0.0
                            rhs = float(np.dot(U[i * m + j], V[jp * p + k] * W[kp * n + ip]))
                            e = abs(lhs - rhs)
                            if e > maxerr:
                                maxerr = e
    return maxerr < tol


def _to_list(M):
    return [[int(round(x)) for x in row] for row in M]


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _kronecker_4x4x5(n, m, p):
    """Build the rank-65 decomposition: Kronecker Strassen (49) for k'=0..3
       plus a 16-term schoolbook strip for the 5th output column k'=4."""
    U2, V2, W2 = _strassen_2x2()

    U4 = np.zeros((16, 49), dtype=int)
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    U4[i * 4 + j, c] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]

    V4 = np.zeros((16, 49), dtype=int)   # (j,k) for k in 0..3
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    V4[j * 4 + k, c] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]

    W4 = np.zeros((16, 49), dtype=int)   # (k',i') for k' in 0..3
    for kp in range(4):
        for ip in range(4):
            k1, k2 = kp // 2, kp % 2
            i1, i2 = ip // 2, ip % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    W4[kp * 4 + ip, c] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]

    # 5th output column (k' = 4): 4x4x1 strip, tensor rank 16.
    U5 = np.zeros((16, 16), dtype=int)
    V5 = np.zeros((20, 16), dtype=int)
    W5 = np.zeros((20, 16), dtype=int)
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U5[i * 4 + j, col] = 1
            V5[j * 5 + 4, col] = 1
            W5[4 * 4 + i, col] = 1   # k'*n + i' = 4*4 + i

    rank = 49 + 16
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((20, rank), dtype=int)
    W = np.zeros((20, rank), dtype=int)
    U[:, 0:49] = U4
    V[:16, 0:49] = V4
    W[:16, 0:49] = W4
    U[:, 49:65] = U5
    V[:, 49:65] = V5
    W[:, 49:65] = W5
    return U, V, W


def _try_newton(rank, n, m, p, max_iter=60, tol=1e-9, deadline=None):
    """Time-boxed Newton search for a rank-r decomposition (fallback only)."""
    n_m, m_p, p_n = n * m, m * p, p * n
    rng = np.random.RandomState(rank * 1000 + 42)
    best = None
    best_err = float('inf')
    for trial in range(3):
        if deadline is not None and time.time() > deadline:
            break
        U = rng.randn(n_m, rank) * 0.1
        V = rng.randn(m_p, rank) * 0.1
        W = rng.randn(p_n, rank) * 0.1
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            S = np.zeros((n, m, p), dtype=np.float64)
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                S += np.einsum('ij,jk->ijk', Uc, Vc) * Wc.T[:, np.newaxis, :]
            R = S - 1.0
            err = np.max(np.abs(R))
            if err < best_err:
                best_err, best = err, (U.copy(), V.copy(), W.copy())
            if err < tol:
                break
            dU = np.zeros_like(U); dV = np.zeros_like(V); dW = np.zeros_like(W)
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                Wc_T = Wc.T
                step1 = R * Wc_T[:, np.newaxis, :]
                dU[:, c] = np.einsum('ijk,jk->ij', step1, Vc).reshape(n_m)
                step2 = R * Wc_T[:, np.newaxis, :]
                dV[:, c] = np.einsum('ijk,ij->jk', step2, Uc).reshape(m_p)
                step3 = R * Vc[np.newaxis, :, :]
                dW[:, c] = np.einsum('ijk,ij->ik', step3, Uc).T.reshape(p_n)
            U -= 0.01 * dU; V -= 0.01 * dV; W -= 0.01 * dW
        if best_err < tol:
            break
    if best is not None and best_err < tol:
        return {"U": [[float(x) for x in r] for r in best[0]],
                "V": [[float(x) for x in r] for r in best[1]],
                "W": [[float(x) for x in r] for r in best[2]], "rank": rank}
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin under 150s

    # Primary: explicit rank-65 Kronecker Strassen + strip construction.
    U, V, W = _kronecker_4x4x5(n, m, p)
    if _verify(U, V, W, n, m, p, tol=1e-9):
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # Fallback 1: bounded numeric search for a lower rank (only if time remains).
    if time.time() < deadline - 20:
        for rank in range(64, 55, -1):
            res = _try_newton(rank, n, m, p, max_iter=40, tol=1e-9, deadline=deadline - 20)
            if res is not None and _verify(res["U"], res["V"], res["W"], n, m, p, tol=1e-8):
                return res

    # Fallback 2: guaranteed schoolbook (rank 80).
    U, V, W = _schoolbook(n, m, p)
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
