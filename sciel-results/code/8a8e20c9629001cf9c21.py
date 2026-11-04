"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # Try ranks from low to high; schoolbook is n*m*p = 60
    for rank in range(36, 61):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result

    # Fallback to schoolbook
    return _schoolbook(n, m, p)


def _try_rank(n, m, p, rank, deadline):
    max_restarts = 40
    max_iter = 400

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = np.random.RandomState(12345 + restart * 7919)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)

        for it in range(max_iter):
            if time.time() > deadline:
                break

            # --- Update U rows (vectorized) ---
            # For each (i, j): solve M^T * u = 1_p where M[k, r] = V[jp+k, r] * W[kn+i, r]
            for i in range(n):
                for j in range(m):
                    M = np.multiply.outer(V[j * p:(j + 1) * p, :],
                                          W[i::n, :])            # (p, rank)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # --- Update V rows (vectorized) ---
            # For each (j, k): solve N^T * v = 1_n where N[i, r] = U[im+j, r] * W[kn+i, r]
            for j in range(m):
                for k in range(p):
                    N_mat = np.multiply.outer(U[j::m, :],
                                              W[k::p, :])            # (n, rank)
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(N_mat.T, np.ones(n), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # --- Update W rows (vectorized) ---
            # For each (k, i): solve P^T * w = 1_m where P[j, r] = U[im+j, r] * V[jp+k, r]
            for k in range(p):
                for i in range(n):
                    P_mat = np.multiply.outer(U[i::n, :],
                                              V[k::p, :])            # (m, rank)
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(P_mat.T, np.ones(m), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # Check convergence every 8 iterations
            if it % 8 == 7:
                max_err = _max_err(U, V, W, n, m, p)
                if max_err < 1e-9:
                    # high-precision polish before rationalizing
                    _polish(U, V, W, n, m, p, deadline)
                    result = _rationalize(U, V, W, n, m, p, rank)
                    if result is not None:
                        return result

    return None


def _max_err(U, V, W, n, m, p):
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _polish(U, V, W, n, m, p, deadline, max_iter=120, tol=1e-12):
    """Final high-precision polish pass with tighter tolerance."""
    for it in range(max_iter):
        if time.time() > deadline:
            break
        for i in range(n):
            for j in range(m):
                M = np.multiply.outer(V[j * p:(j + 1) * p, :], W[i::n, :])
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        for j in range(m):
            for k in range(p):
                N_mat = np.multiply.outer(U[j::m, :], W[k::p, :])
                try:
                    V[j * p + k, :] = np.linalg.lstsq(N_mat.T, np.ones(n), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        for k in range(p):
            for i in range(n):
                P_mat = np.multiply.outer(U[i::n, :], V[k::p, :])
                try:
                    W[k * n + i, :] = np.linalg.lstsq(P_mat.T, np.ones(m), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        if it % 20 == 19 and _max_err(U, V, W, n, m, p) < tol:
            break


def _verify_rational(U_r, V_r, W_r, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _rationalize(U, V, W, n, m, p, rank):
    """Try to snap to small-denominator rationals and verify exact decomposition."""
    for denom in (1, 2, 3, 4, 6):
        U_r = np.round(U * denom) / denom
        V_r = np.round(V * denom) / denom
        W_r = np.round(W * denom) / denom
        if _verify_rational(U_r, V_r, W_r, n, m, p):
            U_list = [[float(U_r[i, j]) for j in range(rank)] for i in range(U_r.shape[0])]
            V_list = [[float(V_r[i, j]) for j in range(rank)] for i in range(V_r.shape[0])]
            W_list = [[float(W_r[i, j]) for j in range(rank)] for i in range(W_r.shape[0])]
            return {"U": U_list, "V": V_list, "W": W_list}
    return None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
