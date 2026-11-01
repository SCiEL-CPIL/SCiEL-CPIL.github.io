"""100-point logarithmic energy minimization on the unit sphere.

Riemannian L-BFGS via exponential map + diverse multi-restart + polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 118.0  # safety margin before 120s kill


# ---------------- energy / gradient (vectorised) ----------------

def _energy(p):
    """sum_{i<j} log(1/|p_i-p_j|) using math.fsum."""
    d = p[:, None, :] - p[None, :, :]
    dsq = np.einsum('ijk,ijk->ij', d, d)
    iu = np.triu_indices(N, 1)
    ds = np.sqrt(np.maximum(dsq[iu], 1e-300))
    return math.fsum(np.log(1.0 / ds).tolist())


def _energy_grad(p):
    """energy and ambient gradient g_i = sum_{j!=i} (p_i - p_j)/|p_i-p_j|^2."""
    d = p[:, None, :] - p[None, :, :]
    dsq = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(dsq, 1.0)
    inv = 1.0 / dsq
    np.fill_diagonal(inv, 0.0)
    g = np.einsum('ij,ijk->ik', inv, d)
    iu = np.triu_indices(N, 1)
    ds = np.sqrt(np.maximum(dsq[iu], 1e-300))
    e = math.fsum(np.log(1.0 / ds).tolist())
    return e, g


def _project(p):
    n = np.linalg.norm(p, axis=1, keepdims=True)
    n = np.maximum(n, 1e-15)
    return p / n


# ---------------- Riemannian L-BFGS via exponential map ----------------

def _exp_map(p, v):
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    safe = np.maximum(theta, 1e-12)
    return (np.sin(safe) / safe) * v + np.cos(safe) * p


def _retr(p, v):
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    safe = np.maximum(theta, 1e-12)
    return (1.0 - 1.0 / (2.0 * np.tan(safe / 2.0))) * v + p


def _riem_lbfgs(p0, budget, maxiter=3000, m=12):
    """Riemannian L-BFGS on the sphere using the exponential map."""
    t0 = time.time()
    p = p0.copy()
    e, g = _energy_grad(p)
    g = g - (g * p).sum(1, keepdims=True) * p
    H = np.eye(3)
    s = np.zeros((m, N, 3)); y = np.zeros((m, N, 3))
    ys = np.zeros(m); yy = np.zeros(m)
    idx = 0; cnt = 0
    best = e; bestp = p.copy()
    for it in range(maxiter):
        if time.time() - t0 > budget:
            break
        # two-loop recursion
        q = g.copy()
        alpha = np.zeros(m)
        for j in range(cnt - 1, -1, -1):
            jj = j % m
            alpha[jj] = ys[jj] / yy[jj] * np.einsum('ij,ij->', s[jj], q)
            q = q - alpha[jj] * y[jj]
        r = H @ q.reshape(-1).reshape(N, 3)
        for j in range(cnt):
            jj = j % m
            beta = ys[jj] / yy[jj] * np.einsum('ij,ij->', y[jj], r)
            r = r + (alpha[jj] - beta) * s[jj]
        d = -r
        # backtracking line search
        step = 1.0
        cur_e = e
        for _ in range(30):
            np_ = _exp_map(p, step * d)
            ne, ng = _energy_grad(np_)
            if ne < cur_e - 1e-4 * step * np.einsum('ij,ij->', g, d):
                break
            step *= 0.5
        else:
            break
        if ne >= e - 1e-16:
            # accept if it's not worse
            if ne < e:
                pass
            else:
                break
        # update
        s_new = (np_ - p) / step
        y_new = (ng - g) - ((ng - g) * p).sum(1, keepdims=True) * p
        yyv = np.einsum('ij,ij->', s_new, y_new)
        p = np_; g = ng; e = ne
        if e < best:
            best = e; bestp = p.copy()
        if yyv > 1e-14:
            s[idx] = s_new; y[idx] = y_new; ys[idx] = np.einsum('ij,ij->', s_new, y_new)
            yy[idx] = yyv; idx = (idx + 1) % m; cnt = min(cnt + 1, m)
        if np.linalg.norm(g) < 1e-13:
            break
    return bestp, best


# ---------------- initializers ----------------

def _fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return _project(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def _random(n, seed):
    rng = np.random.RandomState(seed)
    return _project(rng.randn(n, 3))


def _rings(n, rings, seed):
    rng = np.random.RandomState(seed)
    pts = []
    for k in range(rings):
        z = -1.0 + 2.0 * (k + 0.5) / rings
        r = np.sqrt(max(1.0 - z * z, 0.0))
        n_r = n // rings + (1 if k < n % rings else 0)
        phis = (2.0 * np.pi * np.arange(n_r) / n_r) + rng.uniform(0, 2 * np.pi)
        pts.append(np.column_stack([r * np.cos(phis), r * np.sin(phis),
                                     np.full(n_r, z)]))
    return _project(np.vstack(pts))


def _perturbed(base, amp, seed):
    rng = np.random.RandomState(seed)
    return _project(base + amp * rng.randn(N, 3))


# ---------------- construct ----------------

def construct():
    t0 = time.time()
    bestp = _fibonacci(N)
    best = _energy(bestp)

    def try_opt(p, budget):
        nonlocal bestp, best
        if time.time() - t0 > TIME_LIMIT - 6.0:
            return
        try:
            np_, ne = _riem_lbfgs(p, budget)
        except Exception:
            return
        if ne < best:
            best = ne; bestp = np_

    # Phase 1: primary from Fibonacci
    try_opt(_fibonacci(N), 12.0)

    # Phase 2: diverse restarts
    rng = np.random.RandomState(7)
    start = time.time()
    k = 0
    while time.time() - t0 < TIME_LIMIT - 25.0:
        k += 1
        seed = 1000 + k
        if k % 6 == 1:
            p = _random(N, seed)
        elif k % 6 == 2:
            p = _rings(N, 4, seed)
        elif k % 6 == 3:
            p = _rings(N, 5, seed)
        elif k % 6 == 4:
            p = _perturbed(_fibonacci(N), 0.12, seed)
        elif k % 6 == 5:
            p = _perturbed(bestp, 0.08, seed)
        else:
            p = _perturbed(_fibonacci(N), 0.3, seed)
        budget = max(2.0, min(8.0, (TIME_LIMIT - 25.0) - (time.time() - t0)))
        try_opt(p, budget)

    # Phase 3: polish the best with tight tolerance
    if time.time() - t0 < TIME_LIMIT - 8.0:
        try_opt(bestp, TIME_LIMIT - 8.0 - (time.time() - t0))

    # Final validation
    bestp = _project(bestp)
    d = bestp[:, None, :] - bestp[None, :, :]
    dsq = np.einsum('ijk,ijk->ij', d, d)
    iu = np.triu_indices(N, 1)
    ds = np.sqrt(dsq[iu])
    if np.any(ds < 1e-9):
        rng = np.random.RandomState(0)
        for (i, j) in zip(*iu):
            if dsq[i, j] < 1e-18:
                bestp[i] = _project(bestp[i] + 1e-4 * rng.randn(3))
    bestp = _project(bestp)
    if not np.isfinite(_energy(bestp)):
        bestp = _fibonacci(N)
    return bestp
