"""Logarithmic energy minimization of 100 points on the unit sphere.
Gradient-based (L-BFGS-B on sphere constraints) with many randomized restarts
and a final Newton polish, all under an explicit wall-clock deadline guard."""
import time
import math
import numpy as np

N = 100
DEADLINE = 132.0          # safety margin well under the 150 s kill
_DEADLINE = time.monotonic() + DEADLINE


def _time_left():
    return DEADLINE - (time.monotonic() - _DEADLINE)


def _fibonacci_start(seed):
    rng = np.random.default_rng(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0.0, 2.0 * np.pi)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _rand_start(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _energy_and_grad(x):
    """Return (E, tangent_gradient) for E = sum_{i<j} log(1/||x_i-x_j||)."""
    diff = x[:, None, :] - x[None, :, :]            # (N,N,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-18)
    d = np.sqrt(d2)
    E = -0.5 * math.fsum(np.log(d).ravel().tolist())
    # dE/dx_i = sum_{j!=i} (x_i - x_j)/||x_i-x_j||^2
    w = np.zeros_like(d2)
    np.fill_diagonal(w, 0.0)
    w /= d2
    grad = np.einsum('ij,ij k->k', w, diff)          # (N,3)
    # project onto tangent space
    g = grad - np.einsum('nk,n->nk', grad, x) * x
    return E, g


def _lbfgs(x0):
    try:
        res = scipy.optimize.minimize(
            lambda v: _energy_and_grad(v.reshape(N, 3))[0],
            x0.ravel(),
            jac=lambda v: _energy_and_grad(v.reshape(N, 3))[1].ravel(),
            method='L-BFGS-B',
            bounds=[(-1.0, 1.0)] * (3 * N),
            options={'maxiter': 400, 'gtol': 1e-10, 'ftol': 1e-14, 'maxfun': 400},
        )
        return res.x.reshape(N, 3), res.fun
    except Exception:
        return x0, _energy_and_grad(x0)[0]


def _newton_polish(x0, iters=60):
    """Newton's method on the sphere using the tangent Hessian."""
    x = x0.copy()
    n = x.shape[0]
    for _ in range(iters):
        if _time_left() < 1.5:
            break
        diff = x[:, None, :] - x[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-18)
        d = np.sqrt(d2)
        w = np.zeros_like(d2)
        np.fill_diagonal(w, 0.0)
        w /= d2
        grad = np.einsum('ij,ij k->k', w, diff)
        g = grad - np.einsum('nk,n->nk', grad, x) * x

        # tangent Hessian H_ij = w_ij (I - x_i x_j^T)
        H = w[:, :, None, None] * (np.eye(3) - x[:, None, None, :] * x[None, :, None, :])
        # build full (3N,3N) Hessian, solve, project step to tangent
        A = np.zeros((3 * n, 3 * n))
        A[:3 * n:3, :3 * n:3] = 0.0
        for i in range(n):
            A[3 * i:3 * i + 3, 3 * i:3 * i + 3] = H[i, i]
            for j in range(i + 1, n):
                A[3 * i:3 * i + 3, 3 * j:3 * j + 3] = H[i, j]
                A[3 * j:3 * j + 3, 3 * i:3 * i + 3] = H[i, j]
        gv = g.ravel()
        try:
            step = np.linalg.solve(A + 1e-12 * np.eye(3 * n), -gv)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(A + 1e-12 * np.eye(3 * n), -gv, rcond=None)[0]
        step = step.reshape(n, 3)
        step -= np.einsum('nk,n->nk', step, x) * x
        # damped line search
        best = None
        E0 = _energy_and_grad(x)[0]
        for t in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
            xn = x + t * step
            xn /= np.linalg.norm(xn, axis=1, keepdims=True)
            En = _energy_and_grad(xn)[0]
            if En < E0:
                best = (t, En, xn)
                break
        if best is None:
            break
        x = best[2]
    return x, _energy_and_grad(x)[0]


def construct():
    import scipy.optimize
    global _lbfgs, _newton_polish, _energy_and_grad
    best_x = _fibonacci_start(0)
    best_e = _energy_and_grad(best_x)[0]

    def _eval(x):
        nonlocal best_x, best_e
        e = _energy_and_grad(x)[0]
        if e < best_e:
            best_x, best_e = x, e

    # phase 1: many cheap randomized restarts with L-BFGS-B
    seed = 0
    while _time_left() > 6.0:
        seed += 1
        if seed <= 25:
            x0 = _fibonacci_start(seed)
        else:
            x0 = _rand_start(seed)
        x1, e1 = _lbfgs(x0)
        _eval(x1)
        if _time_left() < 6.0:
            break

    # phase 2: Newton polish on the current best (and a couple of good starts)
    if _time_left() > 3.0:
        x2, e2 = _newton_polish(best_x)
        _eval(x2)

    x = best_x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    # validity guard
    if not np.all(np.isfinite(x)) or np.any(np.linalg.norm(x, axis=1) < 1e-9):
        x = _fibonacci_start(0)
    return x
