"""Rewritten from scratch: physics-based L-BFGS-B circle packing in [-0.5,0.5]^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 135.0  # safety margin below the 150 s kill


def _min_dist(X):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.sqrt(np.maximum(d2, 0.0)).min()


def _wall(X):
    return 0.5 - np.abs(X).max()


def _score(X):
    return min(_min_dist(X) * 0.5, _wall(X))


def _energy(X, r, alpha):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(np.maximum(d2, 0.0))
    D = r / d
    E = (D ** 2).sum()
    w = np.maximum(0.0, 0.5 - np.abs(X))
    E += alpha * np.sum(1.0 / (w + 1e-9) ** 2)
    return E


def _energy_grad(X, r, alpha):
    n = X.shape[0]
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, 0.0)
    d = np.sqrt(np.maximum(d2, 0.0))
    D = r / d
    D.fill_diagonal(0.0)
    diff = X[:, None, :] - X[None, :, :]
    G = np.zeros_like(X)
    np.einsum('ij,ijk->ik', D ** 2, diff, out=G)
    w = 0.5 - np.abs(X)
    G += alpha * (2.0 / (w + 1e-9) ** 3) * np.sign(X)
    return G


def _grid_init(n):
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n].copy()


def _optimize(X0, r0, alpha, budget):
    X = X0.copy()
    cur = X
    for _ in range(3):
        t0 = time.time()
        res = minimize(_energy, X, jac=_energy_grad, method='L-BFGS-B',
                       args=(r0, alpha),
                       options={'maxiter': 400, 'maxfun': 20000,
                                'maxls': 40, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.x is not None and np.all(np.isfinite(res.x)):
            X = res.x
            cur = X
        if time.time() - t0 > budget or time.time() > DEADLINE:
            break
    return cur


def construct():
    start = time.time()
    r_best = 0.0
    X_best = _grid_init(N)
    r_best = _score(X_best)

    # Coarse anneal: start from a wide radius so forces spread particles out,
    # then shrink the target radius to tighten the packing.
    alphas = [40.0, 10.0, 4.0, 1.5, 0.6, 0.25]
    radii = [0.055, 0.050, 0.045, 0.042, 0.039, 0.0375]
    X = _grid_init(N)

    for r0, alpha in zip(radii, alphas):
        if time.time() > DEADLINE:
            break
        X = _optimize(X, r0, alpha, budget=18.0)
        s = _score(X)
        if s > r_best:
            r_best = s
            X_best = X.copy()

    # Final local polish near the known optimum radius.
    if time.time() < DEADLINE - 10.0:
        for r0 in (0.0368, 0.0366, 0.0364):
            if time.time() > DEADLINE - 8.0:
                break
            X = _optimize(X, r0, 0.15, budget=12.0)
            s = _score(X)
            if s > r_best:
                r_best = s
                X_best = X.copy()

    # Safety: ensure a valid finite array inside the square.
    X_best = np.clip(X_best, -0.5, 0.5)
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    return X_best
