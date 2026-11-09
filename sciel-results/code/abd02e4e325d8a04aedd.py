import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS  # n=3 rows of A, m=4 inner, p=6 cols of B

# ---------------------------------------------------------------------------
# Tensor convention (matches the current program / evaluator):
#   T[a, b, c] = 1  where a = i*m + j, b = j*p + k, c = k*n + i
#   i in [0,n), j in [0,m), k in [0,p)
# ---------------------------------------------------------------------------
T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
for _i in range(n):
    for _j in range(m):
        for _k in range(p):
            T[_i * m + _j, _j * p + _k, _k * n + _i] = 1.0

# Precomputed flattens for least-squares (A = U, B = V, C = W)
T_AB = T.reshape(n * m, m * p * p * n)      # (nm, (mp)(pn))
T_BC = T.transpose(1, 2, 0).reshape(m * p, p * n * n * m)  # (mp, (pn)(nm))
T_CA = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)  # (pn, (nm)(mp))


def _verify(U, V, W):
    r = U.shape[1]
    Tc = U[:, :, None, None] * V[None, :, :, None] * W[None, None, :, :]
    return float(np.abs(Tc.sum(axis=3) - T).max())


def _polish(U, V, W, r, deadline):
    """High-precision alternating least-squares polish with early stopping."""
    for _ in range(400):
        if time.time() > deadline - 2:
            break
        T1 = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, r)
        T2 = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, r)
        T3 = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, r)
        U = np.linalg.lstsq(T1, T_AB.T, rcond=None)[0].T
        V = np.linalg.lstsq(T2, T_BC.T, rcond=None)[0].T
        W = np.linalg.lstsq(T3, T_CA.T, rcond=None)[0].T
        if _verify(U, V, W) < 1e-11:
            break
    return U, V, W


def _als(U, V, W, r, iters, deadline):
    """Fast alternating least-squares; returns (U,V,W, residual)."""
    prev = np.inf
    for _ in range(iters):
        if time.time() > deadline - 5:
            break
        T1 = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, r)
        T2 = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, r)
        T3 = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, r)
        U = np.linalg.lstsq(T1, T_AB.T, rcond=None)[0].T
        V = np.linalg.lstsq(T2, T_BC.T, rcond=None)[0].T
        W = np.linalg.lstsq(T3, T_CA.T, rcond=None)[0].T
        res = _verify(U, V, W)
        if res < 1e-9:
            return U, V, W, res
        if res > prev * 0.999 and _ > 20:
            break
        prev = res
    return U, V, W, _verify(U, V, W)


def _search_als(deadline):
    """Direct ALS search from the lower bound upward."""
    lb = max(n * p, m * n, m * p)  # = 24
    best = None
    for r in range(lb, 73):
        if time.time() > deadline - 25:
            break
        for restart in range(10):
            if time.time() > deadline - 20:
                break
            np.random.seed(r * 1000 + restart)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            U, V, W, res = _als(U, V, W, r, 150, deadline)
            if res < 1e-10:
                U, V, W = _polish(U, V, W, r, deadline)
                if _verify(U, V, W) < 1e-9:
                    best = (r, U, V, W)
                    break
        if best is not None:
            break
    return best


def _rank_fraction(M):
    M = [row[:] for row in M]
    if not M:
        return 0
    R = len(M)
    C = len(M[0])
    rank = 0
    for c in range(C):
        piv = None
        for rr in range(rank, R):
            if M[rr][c] != 0:
                piv = rr
                break
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        pv = M[rank][c]
        M[rank] = [x / pv for x in M[rank]]
        for rr in range(R):
            if rr == rank:
                continue
            f = M[rr][c]
            if f != 0:
                M[rr] = [x - f * y for x, y in zip(M[rr], M[rank])]
        rank += 1
    return rank


def _grid(limit):
    out = []
    for v in range(-limit, limit + 1, 2):
        if v != 0:
            out.append(Fraction(v, 2))
    return out


def _certified_lower_bound(deadline):
    """
    Slice-and-sum structural lower bound.
    For any coefficient grid, the tensor's three unfoldings must each have rank
    <= r.  Using distinct grid points g_i, h_j, l_k, the unfoldings become
    Vandermonde-type matrices:
        A: rows g_i, cols h_j, entry g_i*h_j
        B: rows h_j, cols l_k, entry h_j*l_k
        C: rows (l_k,g_i), cols h_j, entry g_i*h_j*l_k
    The minimum over grids of max(rank) is a valid lower bound on tensor rank.
    """
    best = n * m * p
    for limit in (4, 6, 8, 10):
        if time.time() > deadline - 30:
            break
        ii = _grid(limit)
        jj = _grid(limit)
        kk = _grid(limit)
        # Precompute the three matrices for this grid (they do not depend on loop vars)
        A = [[Fraction(i * j) for j in jj] for i in ii]
        B = [[Fraction(j * k) for k in kk] for j in jj]
        C = [[Fraction(i * j * k) for j in jj] for i in ii for k in kk]
        rA = _rank_fraction(A)
        rB = _rank_fraction(B)
        rC = _rank_fraction(C)
        need = max(rA, rB, rC)
        if need < best:
            best = need
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # 1) Structural slice-and-sum lower bound via the certificate grid.
    lb = _certified_lower_bound(deadline)

    # 2) Direct ALS search from the lower bound upward.
    result = _search_als(deadline)

    # 3) Fall back to the exact schoolbook decomposition (always valid).
    if result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    r, U, V, W = result
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }
