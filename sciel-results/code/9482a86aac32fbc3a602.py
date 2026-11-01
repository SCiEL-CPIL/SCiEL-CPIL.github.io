"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 140.0  # safety margin under the 150s wall-clock kill
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _hex_init():
    """Hexagonal lattice cut to the unit disk, returned as (N,2)."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        pts = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
               for i in range(-k, k + 1) for j in range(-k, k + 1)]
        P = np.array(pts)
        rad = np.hypot(P[:, 0], P[:, 1])
        mask = rad <= 1.0 - spacing / 2.0
        P, rad = P[mask], rad[mask]
        if len(P) >= N:
            return P[np.argsort(rad, kind="stable")[:N]]
        spacing *= 0.98


def _score(P):
    """Raw packing score: min of half the min inter-centre distance and the min margin to the boundary."""
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r = 0.5 * d.min()
    m = 1.0 - np.linalg.norm(P, axis=1).max()
    return min(r, m)


def _softmin_neg(x, beta):
    """Negative soft-min of the constraint values (distances and margins) with temperature beta.

    The soft-min approximates min_i g_i; maximizing it (minimizing its negative) pushes all
    constraints up. As beta -> inf it approaches the true min.
    """
    P = x.reshape(-1, 2)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    margins = 1.0 - np.linalg.norm(P, axis=1)
    g = np.concatenate([d.ravel(), margins])
    g = g[np.isfinite(g)]
    if g.size == 0:
        return 0.0
    b = beta
    gmax = g.max()
    # soft-min = -logsumexp(-g)/b  ; return its negative
    s = np.exp(-b * (g - gmax))
    logsum = np.log(s.sum()) / b
    softmin = gmax - logsum
    return -softmin


def _softmin_neg_grad(x, beta):
    P = x.reshape(-1, 2)
    diff = P[:, None, :] - P[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    margins = 1.0 - np.linalg.norm(P, axis=1)
    g = np.concatenate([d.ravel(), margins])
    g = g[np.isfinite(g)]
    if g.size == 0:
        return 0.0, np.zeros_like(x)
    b = beta
    gmax = g.max()
    e = np.exp(-b * (g - gmax))
    e_sum = e.sum()
    w = e / e_sum  # weights for the soft-min
    softmin = gmax - np.log(e_sum) / b
    # gradient of soft-min w.r.t. g_i = -w_i
    # d(softmin)/dx = sum_i (-w_i) * d(g_i)/dx
    grad = np.zeros_like(x)
    # contributions from pairwise distances
    wd = w[:N * (N - 1)]
    wd2 = wd.reshape(N, N - 1)
    # rebuild index mapping: d.ravel() order is row-major of the (N,N) matrix with diag inf
    # We'll just recompute using a stable loop-free approach on the full matrix.
    # Reconstruct full (N,N) weight matrix.
    W = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            W[i, j] = wd[idx]
            idx += 1
    # softmin over distances part: contribution to grad from distance g_ij = |P_i - P_j|
    # d g_ij / d P_i = (P_i - P_j)/d_ij ; d g_ij / d P_j = (P_j - P_i)/d_ij
    # grad = -sum_ij W_ij * d g_ij/d x  (because d(softmin)/d g_ij = -W_ij)
    # => grad contribution = +sum_ij W_ij * d g_ij/d x
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if d[i, j] > 1e-12:
                v = (P[i] - P[j]) / d[i, j]
                grad[2 * i] += W[i, j] * v[0]
                grad[2 * i + 1] += W[i, j] * v[1]
                grad[2 * j] += W[j, i] * v[0]
                grad[2 * j + 1] += W[j, i] * v[1]
    # margins g_k = 1 - |P_k| ; d g_k/d P_k = -P_k/|P_k|
    wm = w[N * (N - 1):]
    for i in range(N):
        r = np.linalg.norm(P[i])
        if r > 1e-12:
            grad[2 * i] += wm[i] * (-P[i][0] / r)
            grad[2 * i + 1] += wm[i] * (-P[i][1] / r)
    # objective is -softmin, so gradient is negated
    return -softmin, -grad


def _softmin_neg_grad_fast(x, beta):
    """Vectorized gradient of the negative soft-min objective."""
    P = x.reshape(-1, 2)
    diff = P[:, None, :] - P[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    d = np.where(np.eye(N, dtype=bool), np.inf, d)
    margins = 1.0 - np.linalg.norm(P, axis=1)
    g = np.concatenate([d.ravel(), margins])
    finite = np.isfinite(g)
    if not finite.any():
        return 0.0, np.zeros_like(x)
    g = g[finite]
    b = beta
    gmax = g.max()
    e = np.exp(-b * (g - gmax))
    e_sum = e.sum()
    w = e / e_sum
    softmin = gmax - np.log(e_sum) / b
    # Rebuild full weight matrix for distances
    W = np.zeros((N, N))
    W.flat = np.concatenate([w[:N * (N - 1)], np.zeros(0)])
    # place correctly: fill row-major skipping diagonal
    row_w = w[:N * (N - 1)].reshape(N, N - 1)
    for i in range(N):
        W[i, :i] = row_w[i, :i]
        W[i, i + 1:] = row_w[i, i:]
    wm = w[N * (N - 1):]
    grad = np.zeros_like(x)
    # distance contributions
    safe_d = np.where(d > 1e-12, d, 1.0)
    V = diff / safe_d[:, :, None]  # (N,N,2) unit vectors (P_i - P_j)/d
    # grad[2i] += sum_j W[i,j] * V[i,j,0]
    G = np.einsum('ij,ijk->k', W, V)
    grad = G.reshape(-1)
    # margin contributions
    r = np.linalg.norm(P, axis=1)
    safe_r = np.where(r > 1e-12, r, 1.0)
    grad[0::2] += wm * (-P[:, 0] / safe_r)
    grad[1::2] += wm * (-P[:, 1] / safe_r)
    return -softmin, -grad


def _anneal(P):
    """Run the beta-annealing L-BFGS-B schedule, returning the best P found."""
    best_P = P.copy()
    best_score = _score(P)
    betas = [40.0, 120.0, 400.0, 1500.0]
    for beta in betas:
        if _remaining() < 8.0:
            break
        try:
            res = minimize(_softmin_neg_grad_fast, best_P.reshape(-1),
                           method='L-BFGS-B', jac=True,
                           options={'maxiter': 60, 'gtol': 1e-5, 'maxls': 20})
            cand = res.x.reshape(-1, 2)
        except Exception:
            cand = best_P
        # keep candidate only if it is valid and better
        if np.all(np.isfinite(cand)):
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_P = cand
    return best_P, best_score


def construct():
    global _START
    _START = time.time()
    P = _hex_init()
    best_P, best_score = P, _score(P)
    # a couple of perturb-and-optimize restarts if time permits
    rng = np.random.default_rng(0)
    attempts = 0
    while _remaining() > 25.0 and attempts < 3:
        attempts += 1
        P0 = best_P + rng.normal(0.0, 0.004, size=best_P.shape)
        # clamp inside boundary to keep valid
        rad = np.linalg.norm(P0, axis=1)
        P0 = P0 / np.maximum(rad, 1.0)[:, None] * np.minimum(rad, 0.999)[:, None]
        cand, s = _anneal(P0)
        if s > best_score:
            best_score = s
            best_P = cand
    # final polish with high beta
    if _remaining() > 6.0:
        cand, s = _anneal(best_P)
        if s > best_score:
            best_score = s
            best_P = cand
    # ensure strictly inside and finite
    rad = np.linalg.norm(best_P, axis=1)
    best_P = best_P / np.maximum(rad, 1.0)[:, None] * np.minimum(rad, 0.9999)[:, None]
    return best_P
# EVOLVE-BLOCK-END
