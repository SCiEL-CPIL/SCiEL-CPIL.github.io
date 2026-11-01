"""Combined: Halton/Fibonacci init + Riemannian CG (exp-map, PR beta) + BFGS polish, deadline-guarded."""
import time
import math
import numpy as np

N = 150
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds before the hard limit


def _log_energy(X):
    """Sum over i<j of log(1/d) with chordal distances, via math.fsum."""
    n = X.shape[0]
    terms = []
    for i in range(n):
        d = X[i] - X[i + 1:]
        dist = np.sqrt(np.sum(d * d, axis=1))
        terms.extend(math.log(1.0 / di) for di in dist)
    return math.fsum(terms)


def _energy_and_grad(X):
    """Energy and tangent-projected gradient for sum_{i<j} log(1/|Xi-Xj|)."""
    n = X.shape[0]
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    # energy (upper triangle)
    E = 0.5 * np.sum(np.log(1.0 / dist))
    # ambient gradient: sum_{j != i} (Xi - Xj)/|Xi-Xj|^2
    inv = np.zeros_like(dist)
    inv[dist > 0] = 1.0 / (dist[dist > 0] ** 2)
    np.fill_diagonal(inv, 0.0)
    G = np.sum(diff * inv[:, :, None], axis=1)
    # project to tangent space
    proj = G - np.sum(G * X, axis=1, keepdims=True) * X
    return E, proj


def _project_to_sphere(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < 1e-15] = 1.0
    return X / nrm


def _exp_map(X, d, alpha):
    """Exponential map on unit sphere: exp_x(alpha*d) for d in tangent space."""
    # d is (N,3) tangent; alpha scalar
    # exp_x(v) = x*cos(||v||) + (v/||v||)*sin(||v||)
    v = alpha * d
    nv = np.linalg.norm(v, axis=1, keepdims=True)
    mask = nv > 1e-15
    out = X.copy()
    if np.any(mask):
        cos_a = np.cos(nv[mask])
        sin_a = np.sin(nv[mask])
        v_hat = v[mask] / nv[mask]
        out[mask] = X[mask] * cos_a + v_hat * sin_a
    return _project_to_sphere(out)


def _halton_init(num_dims, skip=1):
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    pts = []
    for i in range(skip, skip + num_dims):
        pts.append([halton(i, b) for b in (2, 3, 5)])
    pts = np.array(pts)
    i = np.arange(num_dims)
    z = 1.0 - 2.0 * (i + 0.5) / num_dims
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 2.0 * np.pi * pts[:, 0] * 0.01
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _lbfgs_refine(X, time_budget):
    """L-BFGS-B on ambient coords with periodic re-projection."""
    from scipy.optimize import minimize
    best = X.copy()
    best_E = _log_energy(best)
    t0 = time.time()

    def fun(g):
        Xc = _project_to_sphere(g.reshape(N, 3))
        E, _ = _energy_and_grad(Xc)
        return E

    def jac(g):
        Xc = _project_to_sphere(g.reshape(N, 3))
        _, G = _energy_and_grad(Xc)
        return G.reshape(-1)

    res = minimize(fun, X.reshape(-1), jac=jac, method='L-BFGS-B',
                   options={'maxiter': 300, 'maxfun': 600, 'maxls': 30,
                            'ftol': 1e-14, 'gtol': 1e-10})
    Xr = _project_to_sphere(res.x.reshape(N, 3))
    Er = _log_energy(Xr)
    if Er < best_E:
        best = Xr
        best_E = Er
    return best, best_E


def _riemannian_cg(X, time_budget, max_iter=3000):
    """Riemannian CG on sphere using exponential map and Polak-Ribiere beta."""
    best = X.copy()
    best_E = _log_energy(best)
    t0 = time.time()
    Xc = X.copy()
    E, G = _energy_and_grad(Xc)
    d = -G.copy()
    prev_G = G.copy()
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        # backtracking line search with exponential map
        alpha = 0.5
        prev_E = E
        Xn = Xc
        En = E
        Gn = G
        for ls in range(30):
            Xn = _exp_map(Xc, d, alpha)
            En, Gn = _energy_and_grad(Xn)
            if En < prev_E:
                break
            alpha *= 0.5
        if En >= prev_E:
            # no improvement; try a tiny step
            Xn = _exp_map(Xc, d, 1e-6)
            En, Gn = _energy_and_grad(Xn)
            if En >= prev_E:
                if En < best_E:
                    best, best_E = Xn, En
                break
        Xc = Xn
        E = En
        if E < best_E:
            best = Xc.copy()
            best_E = E
        # Polak-Ribiere beta
        beta_num = np.sum(Gn * (Gn - prev_G))
        beta_den = np.sum(prev_G * prev_G)
        beta = max(0.0, beta_num / beta_den) if beta_den > 1e-30 else 0.0
        d = -Gn + beta * d
        # project d to tangent
        d = d - np.sum(d * Xc, axis=1, keepdims=True) * Xc
        prev_G = Gn
        if np.linalg.norm(Gn) < 1e-11:
            break
    return best, best_E


def _bfgs_refine(X, time_budget, max_iter=500):
    """BFGS quasi-Newton on tangent space with exponential-map line search."""
    best = X.copy()
    best_E = _log_energy(best)
    t0 = time.time()
    Xc = X.copy()
    E, G = _energy_and_grad(Xc)
    d = -G.copy()
    # BFGS inverse Hessian approximation B (tangent, per-point block diagonal approx)
    # Use a global scalar-free BFGS on the flattened tangent vector
    n_flat = N * 3
    B = np.eye(n_flat)
    prev_G_flat = G.reshape(-1).copy()
    prev_d_flat = d.reshape(-1).copy()
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        # line search
        alpha = 0.5
        prev_E = E
        Xn = Xc
        En = E
        Gn = G
        for ls in range(25):
            Xn = _exp_map(Xc, d, alpha)
            En, Gn = _energy_and_grad(Xn)
            if En < prev_E:
                break
            alpha *= 0.5
        if En >= prev_E:
            Xn = _exp_map(Xc, d, 1e-7)
            En, Gn = _energy_and_grad(Xn)
            if En >= prev_E:
                if En < best_E:
                    best, best_E = Xn, En
                break
        Xc = Xn
        E = En
        if E < best_E:
            best = Xc.copy()
            best_E = E
        # BFGS update
        G_flat = Gn.reshape(-1)
        d_flat = d.reshape(-1)
        s = Xc.reshape(-1) - _project_to_sphere(Xc + 0.0).reshape(-1)  # placeholder
        # compute s in tangent: s = exp step
        v = alpha * d
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        mask = nv > 1e-15
        s_vec = np.zeros_like(v)
        if np.any(mask):
            cos_a = np.cos(nv[mask])
            sin_a = np.sin(nv[mask])
            v_hat = v[mask] / nv[mask]
            s_vec[mask] = Xc[mask] * cos_a + v_hat * sin_a - Xc[mask]
        s_flat = s_vec.reshape(-1)
        y_flat = G_flat - prev_G_flat
        sy = np.dot(s_flat, y_flat)
        if sy > 1e-12:
            Hy = np.dot(B, y_flat)
            Hyy = np.dot(Hy, y_flat)
            # BFGS update: B_new = B + (s s^T)/(s^T y) - (B y y^T B)/(y^T B y)
            B = B + np.outer(s_flat, s_flat) / sy - np.outer(Hy, Hy) / max(Hyy, 1e-15)
        # new direction
        d = -np.dot(B, G_flat).reshape(N, 3)
        # project to tangent
        d = d - np.sum(d * Xc, axis=1, keepdims=True) * Xc
        prev_G_flat = G_flat
        prev_d_flat = d_flat
        if np.linalg.norm(Gn) < 1e-11:
            break
    return best, best_E


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    inits = []
    inits.append(_fibonacci_init())
    inits.append(_halton_init(N))
    rng = np.random.default_rng(12345)
    inits.append(_project_to_sphere(_fibonacci_init() + 0.05 * rng.standard_normal((N, 3))))
    inits.append(_project_to_sphere(_fibonacci_init() + 0.2 * rng.standard_normal((N, 3))))
    rng2 = np.random.default_rng(99999)
    inits.append(_project_to_sphere(rng2.standard_normal((N, 3))))

    best = inits[0]
    best_E = _log_energy(best)

    for k, X0 in enumerate(inits):
        if time.time() > deadline:
            break
        rem = max(0.5, deadline - time.time())
        # Riemannian CG (primary)
        Xb, Eb = _riemannian_cg(X0, max(0.5, rem * 0.6))
        if Eb < best_E:
            best, best_E = Xb, Eb
        if time.time() > deadline:
            break
        # L-BFGS-B (secondary)
        Xa, Ea = _lbfgs_refine(X0, max(0.5, deadline - time.time()))
        if Ea < best_E:
            best, best_E = Xa, Ea

    # Final polish: BFGS on best
    if time.time() < deadline:
        Xp, Ep = _bfgs_refine(best, max(0.5, deadline - time.time()))
        if Ep < best_E:
            best, best_E = Xp, Ep

    # One more CG polish on the very best
    if time.time() < deadline:
        Xp, Ep = _riemannian_cg(best, max(0.5, deadline - time.time()))
        if Ep < best_E:
            best, best_E = Xp, Ep

    best = _project_to_sphere(best)
    if not np.all(np.isfinite(best)):
        best = _fibonacci_init()
    return best
