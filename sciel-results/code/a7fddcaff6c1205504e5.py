# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_BUDGET = 600.0
_MARGIN = 45.0


def _union_1d(p, N):
    """Exact 1-D union length of the N sheared unit-width/1-N-height intervals."""
    n = N
    lo = np.empty((n, 2), dtype=np.float64)
    hi = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        base = p[i]
        s = i / n
        lo[i, 0] = base
        lo[i, 1] = base + s
        hi[i, 0] = base + 1.0 / n
        hi[i, 1] = base + s + 1.0 / n
    # events: (x, sign)
    ev = np.empty((2 * n, 2), dtype=np.float64)
    ev[:n, 0] = lo[:, 0]
    ev[:n, 1] = 1.0
    ev[n:2 * n, 0] = hi[:, 0]
    ev[n:2 * n, 1] = -1.0
    order = np.lexsort((ev[:, 1], ev[:, 0]))
    x = ev[order, 0]
    sgn = ev[order, 1]
    # group by identical x, net delta
    gx = np.empty(2 * n, dtype=np.float64)
    gd = np.empty(2 * n, dtype=np.float64)
    m = 0
    i = 0
    E = 2 * n
    while i < E:
        xv = x[i]
        d = 0.0
        while i < E and x[i] == xv:
            d += sgn[i]
            i += 1
        gx[m] = xv
        gd[m] = d
        m += 1
    # sweep
    total = 0.0
    cnt = 0.0
    prev = gx[0]
    for k in range(m):
        if k > 0:
            dx = gx[k] - prev
            if dx > 0.0:
                if cnt > 0.0:
                    total += dx * min(1.0, cnt) / n
        cnt += gd[k]
        prev = gx[k]
    return total


def _score1d(p, N):
    return _union_1d(p, N)


def _score_full(P, N):
    # P shape (N,2): column 0 = base x per i, column 1 = base y per j
    return _union_1d(P[:, 0], N) * _union_1d(P[:, 1], N)


def _keich(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _cmaes(f, x0, sigma, max_iter=1200, pop=None, seed=0):
    """Minimal CMA-ES with adaptive step size and PCA covariance update."""
    rng = np.random.default_rng(seed)
    n = len(x0)
    pop = int(pop) if pop else 4 + int(round(3 * np.log(n)))
    if pop < 4:
        pop = 4
    lam = pop
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = int(np.sum(weights[:mu]) ** 2 / np.sum(weights ** 2))
    c = (4 + mueff / n) / ((n + 4) ** 2 + mueff / n)
    d = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c
    c_sigma = (mueff + 2) / (n + mueff + 5)
    c_cov = (4 - mueff / n - c) / ((n + 1) ** 1.8 + mueff + 5)
    chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
    B = np.eye(n)
    D = np.ones(n)
    C = np.eye(n)
    mean = np.array(x0, dtype=np.float64).copy()
    sigma = float(sigma)
    ps = np.zeros(n)
    pc = np.zeros(n)
    best = None
    best_f = np.inf
    for it in range(max_iter):
        if time.time() > _BUDGET - _MARGIN:
            break
        # sample
        Z = rng.standard_normal((lam, n))
        # C = B diag(D^2) B^T
        Y = Z @ B * D  # (lam, n)
        pop_x = np.empty((lam, n))
        for k in range(lam):
            pop_x[k] = mean + sigma * Y[k]
        fvals = np.empty(lam)
        for k in range(lam):
            fvals[k] = f(pop_x[k])
        order = np.argsort(fvals)
        # track best
        fmin = fvals[order[0]]
        if fmin < best_f:
            best_f = fmin
            best = pop_x[order[0]].copy()
        # weighted mean
        xw = (weights[:, None] * pop_x[order[:mu]]).sum(axis=0)
        # update ps
        ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D * ((xw - mean))))
        if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
            pc = (1 - c_cov) * pc + c_cov * (xw - mean) / sigma
        # update C
        C = (1 - c_cov) * C
        delta_h = (1 - c_cov) * (2 - c_cov) * (pc @ pc - np.eye(n))
        C += c_cov * (np.outer(xw - mean, xw - mean) / sigma ** 2) + delta_h
        # ensure PSD
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        C = (eigvecs * eigvals) @ eigvecs.T
        # update B, D
        eigvals, B = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        D = np.sqrt(eigvals)
        # update sigma
        sigma *= np.exp(c_sigma / d * (np.linalg.norm(ps) / chi_n - 1))
        if sigma > 1e6:
            sigma = 1e6
        if sigma < 1e-8:
            sigma = 1e-8
        mean = xw
        if np.std(fvals) < 1e-14 * (1 + abs(fmin)):
            break
    if best is None:
        best = mean
    return best, best_f


def _hill_climb(f, x, base_step, iters, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    best = x.copy()
    best_f = f(best)
    step = base_step
    for it in range(iters):
        if time.time() > _BUDGET - _MARGIN:
            break
        d = rng.standard_normal(n)
        cand = best + step * d
        fc = f(cand)
        if fc < best_f:
            best = cand
            best_f = fc
            step = min(step * 1.15, 1.0)
        else:
            step *= 0.85
    return best, best_f


def _build_positions(px, py, N):
    px = np.asarray(px, dtype=np.float64).ravel()[:N]
    py = np.asarray(py, dtype=np.float64).ravel()[:N]
    P = np.stack([px, py], axis=1)  # (N,2)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = px[i_flat]
    base_y = py[j_flat]
    return np.stack([base_x, base_y], axis=1)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    keich = _keich(N)
    # initial guess: keich
    px0 = keich.copy()
    py0 = keich.copy()
    P0 = _build_positions(px0, py0, N)
    V0 = _score_full(P0, N)

    # ---- Stage 1: optimize each 1-D axis independently via CMA-ES ----
    def f_x(px):
        return _union_1d(px, N)

    def f_y(py):
        return _union_1d(py, N)

    px, fx = _cmaes(f_x, px0, sigma=0.5, max_iter=900, seed=1)
    py, fy = _cmaes(f_y, py0, sigma=0.5, max_iter=900, seed=2)

    P = _build_positions(px, py, N)
    V = _score_full(P, N)
    if V <= 0 or not np.all(np.isfinite(P)):
        P = P0
        V = V0

    # ---- Stage 2: joint fine-tuning of (px, py) via CMA-ES on full V ----
    def f_joint(z):
        return _score_full(_build_positions(z[:N], z[N:], N), N)

    z0 = np.concatenate([px, py])
    z, fz = _cmaes(f_joint, z0, sigma=0.05, max_iter=700, seed=3)
    Pj = _build_positions(z[:N], z[N:], N)
    Vj = _score_full(Pj, N)
    if Vj < V and np.all(np.isfinite(Pj)) and Vj > 0:
        P = Pj
        V = Vj

    # ---- Stage 3: local hill-climb polish on full 128-dim ----
    def f_full(z):
        return _score_full(_build_positions(z[:N], z[N:], N), N)

    zc = np.concatenate([P[:, 0][np.arange(0, 64, N)], P[:, 1][:8]])
    # actually zc should be the base x per i and base y per j
    zc = np.concatenate([px[:N], py[:N]])
    # rebuild from current best P
    zc = np.concatenate([P[:, 0][::N][:N], P[:, 1][:N]])
    zc, fzc = _hill_climb(f_full, zc, base_step=0.02, iters=4000, seed=4)
    Ph = _build_positions(zc[:N], zc[N:], N)
    Vh = _score_full(Ph, N)
    if Vh < V and np.all(np.isfinite(Ph)) and Vh > 0:
        P = Ph
        V = Vh

    # sanity: ensure finite and box area > 1e-9
    if not np.all(np.isfinite(P)):
        P = P0
    return P


# EVOLVE-BLOCK-END
