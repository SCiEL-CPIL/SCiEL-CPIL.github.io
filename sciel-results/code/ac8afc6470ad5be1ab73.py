"""Rewritten program: Chebyshev spectral collocation + fast vectorized search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 84.0  # safety margin before 90 s wall-clock


def _ok():
    return time.time() < _DEADLINE


def _E_cpl(a, w0, wa):
    """E(a) = H(a)/H0 for CPL model in flat universe (vectorized over a)."""
    x = 1.0 - a
    base = 1.0 + x * (w0 + wa * x)
    return np.exp(-3.0 * np.log(a) + 1.5 * np.log(base))


def _comoving_true(z, w0, wa, m=256):
    """High-accuracy comoving distance integral via Gauss-Legendre on [0,1]."""
    xs, ws = np.polynomial.legendre.leggauss(m)
    zz = z * xs
    a = 1.0 / (1.0 + zz)
    vals = 1.0 / _E_cpl(a, w0, wa)
    return z * np.sum(ws * vals)


def _comoving_quad(nodes, weights, z, w0, wa):
    """8-sample quadrature of the comoving distance."""
    a = 1.0 / (1.0 + z * nodes)
    return z * np.sum(weights / _E_cpl(a, w0, wa))


def _worst(nodes, weights, cosmologies, z_arr):
    """Worst relative error (fraction) across all cosmologies and redshifts."""
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    z = np.asarray(z_arr, dtype=float)
    worst = 0.0
    for w0, wa in cosmologies:
        a = 1.0 / (1.0 + np.outer(z, nodes))          # [nz, n]
        q = z[:, None] * (1.0 / _E_cpl(a, w0, wa)).T @ weights  # [nz]
        true = np.array([_comoving_true(zi, w0, wa) for zi in z])
        denom = np.abs(true)
        denom = np.where(denom > 1e-12, denom, 1.0)
        err = np.abs(q - true) / denom
        w = float(err.max())
        if w > worst:
            worst = w
    return worst


def _chebyshev_rule():
    """Chebyshev spectral collocation: exact integration of a truncated Chebyshev
    series of the normalized integrand f(u)=1/E(a(u)) on [0,1]."""
    n = 8
    # Chebyshev points of the 2nd kind on [0,1]
    j = np.arange(n)
    u = 0.5 * (1.0 - np.cos(np.pi * j / (n - 1)))
    # Map u in [0,1] to Chebyshev variable x in [-1,1]
    x = 2.0 * u - 1.0

    # Build Chebyshev basis V[k,i] = T_k(x_i)
    V = np.zeros((n, n))
    V[0, :] = 1.0
    V[1, :] = x
    for k in range(2, n):
        V[k, :] = 2.0 * x * V[k - 1, :] - V[k - 2, :]

    # Reference integrand: use a representative CPL model to shape the rule.
    w0r, war = -1.0, 0.0
    a_ref = 1.0 / (1.0 + u)
    f = 1.0 / _E_cpl(a_ref, w0r, war)

    # Chebyshev coefficients c_k = (2/(n-1)) * sum_i f_i T_k(x_i)  (with end corrections)
    c = (2.0 / (n - 1)) * (V @ f)
    c[0] *= 0.5
    c[-1] *= 0.5

    # Exact integral of T_k on [0,1] = int_{-1}^{1} T_k(t) dt / 2
    # int_{-1}^{1} T_k(t) dt = 0 for odd k; for even k: 2*(-1)^{k/2} * k^2/(k^2-1)
    I = np.zeros(n)
    for k in range(n):
        if k % 2 == 1:
            I[k] = 0.0
        else:
            if k == 0:
                I[k] = 2.0
            else:
                I[k] = 2.0 * ((-1.0) ** (k // 2)) * (k * k) / (k * k - 1.0)
    I *= 0.5

    # Weights: solve V^T w = I  (n equations, n unknowns)
    try:
        w = np.linalg.solve(V.T, I)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(V.T, I, rcond=None)[0]

    # Ensure positivity and normalization
    w = np.abs(w)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        w = np.full(n, 1.0 / n)
    else:
        w /= s

    u = np.sort(u)
    return u, w


def _nelder_mead(x0, f, bounds, max_iter, tol):
    """Simple Nelder-Mead with bounds clipping; returns best point and value."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    step = 0.05
    for d in range(n):
        pt = simplex[0].copy()
        pt[d] += step
        simplex.append(pt)
    fvals = [f(p) for p in simplex]
    best_iter = 0
    for _ in range(max_iter):
        if not _ok():
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if best_iter < 200 and fvals[0] < 0.0:
            pass
        if (fvals[-1] - fvals[0]) < tol:
            break
        # centroid of all but worst
        cen = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        fworst = fvals[-1]
        # reflection
        alpha = 1.0
        xr = cen + alpha * (cen - worst)
        fr = f(np.clip(xr, bounds[0], bounds[1]))
        if fr < fvals[0]:
            # expansion
            gamma = 2.0
            xe = cen + gamma * (cen - worst)
            fe = f(np.clip(xe, bounds[0], bounds[1]))
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # contraction
            rho = 0.5
            if fr < fworst:
                xc = cen + rho * (worst - cen)
            else:
                xc = cen - rho * (worst - cen)
            fc = f(np.clip(xc, bounds[0], bounds[1]))
            if fc < min(fr, fworst):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
        if fvals[0] < fvals[best_iter]:
            best_iter = 0
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]


def _optimize():
    n = 8
    # Training cosmologies (w0, wa)
    w0_vals = [-0.5, -0.8, -1.0, -1.2, -1.5]
    wa_vals = [-0.3, -0.1, 0.0, 0.1, 0.3]
    cosmologies = [(w0, wa) for w0 in w0_vals for wa in wa_vals]
    z_arr = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])

    # Start from Chebyshev spectral rule
    nodes, weights = _chebyshev_rule()

    def obj(params):
        # params: [logit nodes (n-1 interior), log weights (n)]
        ln = params[:n - 1]
        lw = params[n - 1:]
        # build strictly increasing nodes in (0,1)
        inner = 1.0 / (1.0 + np.exp(-ln))
        nodes = np.concatenate([[0.0], inner, [1.0]])
        # ensure strict increase
        for i in range(1, n):
            if nodes[i] <= nodes[i - 1] + 1e-7:
                nodes[i] = nodes[i - 1] + 1e-7
        nodes[0] = max(nodes[0], 1e-7)
        nodes[-1] = min(nodes[-1], 1.0 - 1e-7)
        w = np.exp(lw)
        w /= w.sum()
        return _worst(nodes, w, cosmologies, z_arr)

    # Initial parameters from Chebyshev rule
    inner_nodes = nodes[1:-1]
    ln0 = np.log(inner_nodes / (1.0 - inner_nodes))
    lw0 = np.log(np.clip(weights, 1e-12, None))
    x0 = np.concatenate([ln0, lw0])
    lb = np.concatenate([np.full(n - 1, -12.0), np.full(n, -12.0)])
    ub = np.concatenate([np.full(n - 1, 12.0), np.full(n, 12.0)])

    best_x = x0.copy()
    best_f = obj(x0)

    # Nelder-Mead from Chebyshev start
    x_best, f_best = _nelder_mead(x0, obj, (lb, ub), max_iter=400, tol=1e-7)
    if f_best < best_f:
        best_f, best_x = f_best, x_best

    # A few diverse restarts if time remains
    rng = np.random.default_rng(7)
    for _ in range(6):
        if not _ok():
            break
        x_try = x0 + rng.normal(0.0, 0.3, len(x0))
        x_try = np.clip(x_try, lb, ub)
        x_r, f_r = _nelder_mead(x_try, obj, (lb, ub), max_iter=200, tol=1e-7)
        if f_r < best_f:
            best_f, best_x = f_r, x_r

    # Recover nodes/weights from best params
    ln = best_x[:n - 1]
    lw = best_x[n - 1:]
    inner = 1.0 / (1.0 + np.exp(-ln))
    nodes = np.concatenate([[0.0], inner, [1.0]])
    for i in range(1, n):
        if nodes[i] <= nodes[i - 1] + 1e-7:
            nodes[i] = nodes[i - 1] + 1e-7
    nodes[0] = max(nodes[0], 1e-7)
    nodes[-1] = min(nodes[-1], 1.0 - 1e-7)
    w = np.exp(lw)
    w = np.abs(w)
    w /= w.sum()
    return nodes, w


def construct():
    try:
        nodes, weights = _optimize()
    except Exception:
        nodes = (np.arange(8) + 0.5) / 8.0
        weights = np.full(8, 1.0 / 8.0)

    # Final safety normalization
    weights = np.abs(weights)
    weights /= weights.sum()
    nodes = np.sort(np.asarray(nodes, dtype=float))
    for i in range(1, len(nodes)):
        if nodes[i] <= nodes[i - 1]:
            nodes[i] = nodes[i - 1] + 1e-7
    nodes[0] = max(nodes[0], 1e-7)
    nodes[-1] = min(nodes[-1], 1.0 - 1e-7)
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
