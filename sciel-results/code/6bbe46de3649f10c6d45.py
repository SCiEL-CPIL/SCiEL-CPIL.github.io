"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 130.0  # safety margin below the 150s kill


def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Verify sum_c U[u,c]*V[v,c]*W[io,c] == T[io,u,v] for all (io,u,v)."""
    rank = len(U[0])
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                if s != T.get((io, u, v), 0):
                    return False
    return True


def _time_exhausted():
    return time.time() - _START > _DEADLINE


def _try_als(n, m, p, rank, T, max_iters=40, seeds=2):
    """Try alternating least squares to find a rank-r decomposition."""
    T_arr = np.zeros((p * n, n * m, m * p))
    for (io, u, v), val in T.items():
        T_arr[io, u, v] = val

    for seed in range(seeds):
        if _time_exhausted():
            return None
        np.random.seed(seed)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)

        for it in range(max_iters):
            if _time_exhausted():
                return None
            try:
                M = np.array([np.outer(U[:, c], V[:, c]).flatten() for c in range(rank)])
                T_flat = T_arr.reshape(p * n, n * m * m * p)
                W = np.linalg.lstsq(M, T_flat.T, rcond=None)[0].T

                N = np.array([np.outer(W[:, c], V[:, c]).flatten() for c in range(rank)])
                T_flat2 = T_arr.transpose(1, 2, 0).reshape(n * m, m * p * p * n)
                U = np.linalg.lstsq(N, T_flat2.T, rcond=None)[0].T

                P = np.array([np.outer(W[:, c], U[:, c]).flatten() for c in range(rank)])
                T_flat3 = T_arr.transpose(2, 0, 1).reshape(m * p, p * n * n * m)
                V = np.linalg.lstsq(P, T_flat3.T, rcond=None)[0].T
            except Exception:
                return None

        # Check convergence and try rounding to half-integers
        recon = np.einsum('uc,vc,ic->uvw', U, V, W)
        error = np.max(np.abs(recon - T_arr))
        if error < 1e-3:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            recon_r = np.einsum('uc,vc,ic->uvw', U_r, V_r, W_r)
            if np.max(np.abs(recon_r - T_arr)) < 1e-9:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    return None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    best = _schoolbook(n, m, p)

    if _time_exhausted():
        return best

    # Try structured lower-rank searches via ALS
    for rank in [136, 128, 120, 112, 104, 96, 88, 80, 72]:
        if _time_exhausted():
            break
        result = _try_als(n, m, p, rank, T)
        if result is not None:
            if _exact_check(result["U"], result["V"], result["W"], T, n, m, p):
                return result

    # Fallback: schoolbook (guaranteed valid).
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        return best

    return best
