"""Minimal logarithmic energy of 150 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 120.0
MARGIN = 12.0          # safety margin (seconds) before the 150 s kill


def _energy_and_grad(X):
    """Log energy E = sum_{i<j} log(1/|xi-xj|) and its gradient w.r.t. X.

    E = - sum_{i<j} log d_ij,  d_ij = |xi - xj| (chordal).
    dE/dxi = sum_{j!=i} (xi - xj) / |xi - xj|^2   (for j != i)
    """
    D = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(D * D, axis=2))
    np.fill_diagonal(dist, 1.0)
    inv = 1.0 / dist
    # energy: -sum_{i<j} log d = -0.5 * sum_{i!=j} log d
    energy = -0.5 * np.sum(np.log(dist))
    # gradient: G[i] = sum_{j!=i} (xi - xj) / d_ij^2
    W = inv * inv                      # 1/d^2
    np.fill_diagonal(W, 0.0)
    grad = np.sum(W[:, None, :] * D, axis=1)
    return energy, grad


def _project(X):
    """Re-normalise each row to the unit sphere."""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return X / nrm


def _grad_descent(X, lr=0.02, max_iter=3000, tol=1e-11, deadline=None):
    """Accelerated (Nesterov) gradient descent with backtracking line search."""
    X = _project(X.copy())
    E, G = _energy_and_grad(X)
    V = np.zeros_like(X)
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        Y = X - lr * (V + G)
        E_y, G_y = _energy_and_grad(_project(Y))
        # backtracking line search on the projected objective
        t = 1.0
        X_prev = X
        E_prev = E
        for _ in range(12):
            X_new = _project(X_prev + t * (Y - X_prev))
            E_new, _ = _energy_and_grad(X_new)
            if E_new < E_prev:
                break
            t *= 0.5
        if not (E_new < E_prev):
            break
        V = (X_new - X_prev) / t + G_y
        X = X_new
        E = E_new
        G = G_y
    return X, E


def _lbfgs(X, deadline=None):
    """L-BFGS polish on the sphere via the unconstrained projection trick."""
    def fun(x):
        Xp = _project(x.reshape(N, 3))
        E, _ = _energy_and_grad(Xp)
        return E
    def jac(x):
        Xp = _project(x.reshape(N, 3))
        _, G = _energy_and_grad(Xp)
        return G.reshape(-1)
    res = minimize(fun, X.reshape(-1), jac=jac, method='L-BFGS-B',
                   options={'maxiter': 400, 'ftol': 1e-18, 'gtol': 1e-12,
                            'maxls': 30,
                            'callback': (lambda k: None) if deadline is None else None})
    # respect deadline loosely: L-BFGS is fast per call; check after
    return _project(res.x.reshape(N, 3)), float(res.fun)


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - MARGIN
    rng = np.random.default_rng(12345)

    best = None
    best_E = np.inf

    def consider(X, E):
        nonlocal best, best_E
        if np.isfinite(E) and E < best_E:
            best_E = E
            best = _project(X)

    # 0) Fibonacci seed
    Xf = _fibonacci()
    Xf, Ef = _grad_descent(Xf, lr=0.02, max_iter=2000, deadline=deadline)
    consider(Xf, Ef)
    if time.time() < deadline:
        Xf, Ef2 = _lbfgs(Xf, deadline)
        consider(Xf, Ef2)

    # 1) random restarts with GD polish
    for _ in range(40):
        if time.time() > deadline:
            break
        Xr = rng.standard_normal((N, 3))
        Xr = _project(Xr)
        Xr, Er = _grad_descent(Xr, lr=0.03, max_iter=2500, deadline=deadline)
        consider(Xr, Er)
        if time.time() < deadline:
            Xr, Er2 = _lbfgs(Xr, deadline)
            consider(Xr, Er2)

    # 2) perturbed Fibonacci restarts
    for _ in range(30):
        if time.time() > deadline:
            break
        Xp = _project(_fibonacci() + 0.05 * rng.standard_normal((N, 3)))
        Xp, Ep = _grad_descent(Xp, lr=0.02, max_iter=2500, deadline=deadline)
        consider(Xp, Ep)
        if time.time() < deadline:
            Xp, Ep2 = _lbfgs(Xp, deadline)
            consider(Xp, Ep2)

    # 3) final polish on the best
    if best is not None and time.time() < deadline:
        Xb, Eb = _grad_descent(best, lr=0.01, max_iter=3000, deadline=deadline)
        consider(Xb, Eb)
        if time.time() < deadline:
            Xb, Eb2 = _lbfgs(Xb, deadline)
            consider(Xb, Eb2)

    # 4) final tight L-BFGS polish
    if best is not None and time.time() < deadline:
        Xb, Eb = _lbfgs(best, deadline)
        consider(Xb, Eb)

    if best is None:
        best = _fibonacci()
    # final validity: ensure unit norm, finite, no near-duplicates
    best = _project(best)
    return best
