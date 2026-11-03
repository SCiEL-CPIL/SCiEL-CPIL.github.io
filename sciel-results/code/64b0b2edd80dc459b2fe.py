"""Minimal logarithmic energy of 100 points on the sphere.

Strategy: start from a jittered Fibonacci spiral, run a tangent-plane
gradient descent with backtracking line search (vectorised), then a
high-precision L-BFGS-B polish in (theta, phi) coordinates.  A wall-clock
deadline guard returns the best configuration found so far.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # seconds; 5 s safety margin under the 150 s kill.
START_TIME = time.time()


def _remaining():
    return DEADLINE - (time.time() - START_TIME)


def _energy(x):
    """Sum_{i<j} log(1/|x_i - x_j|) using chordal distances."""
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    # log(1/d) = -log(d); only upper triangle
    iu = np.triu_indices(N, 1)
    return -np.sum(np.log(d[iu]))


def _energy_grad(x):
    """Energy and gradient (Euclidean, not yet projected)."""
    diff = x[:, None, :] - x[None, :, :]          # (N, N, 3)
    d = np.linalg.norm(diff, axis=2)              # (N, N)
    np.fill_diagonal(d, 1.0)
    # log(1/d) = -log d ; dE/dx_i = sum_{j != i} (x_i - x_j)/|x_i - x_j|
    # (because d(-log d)/dx_i = -(x_i-x_j)/d * (-1/d) = (x_i-x_j)/d^2 * ... let me recompute)
    # d = |x_i - x_j|,  f = -log d,  df/dx_i = -(1/d) * (x_i - x_j)/d = -(x_i-x_j)/d^2
    # Wait: log(1/d) = -log d.  d(-log d)/dx_i = -(1/d)*(x_i - x_j)/d = -(x_i - x_j)/d^2.
    # But the energy is sum_{i<j} log(1/|x_i-x_j|).  For pair (i,j) with i<j:
    #   dE/dx_i = -(x_i - x_j)/d^2,  dE/dx_j = +(x_i - x_j)/d^2.
    # So dE/dx_i = sum_{j != i} sign * (x_i - x_j)/d^2 ... let's just do it symmetrically.
    # For each pair (i,j), contribution to grad at i is (x_i - x_j)/d^2 * (-1)
    # and at j is (x_j - x_i)/d^2 * (-1) = (x_i - x_j)/d^2.
    # Actually: f(i,j) = -log|xi-xj|.  grad_xi f = -(xi-xj)/|xi-xj|^2.
    # grad_xj f = +(xi-xj)/|xi-xj|^2 = -(xj-xi)/|xi-xj|^2.
    # So grad_xi = -sum_{j!=i} (xi-xj)/d_ij^2,  grad_xj = +sum_{i!=j} (xi-xj)/d_ij^2.
    # Equivalently: grad_xi = sum_{j!=i} (xj-xi)/d_ij^2.
    inv_d2 = np.zeros_like(d)
    inv_d2[d > 1e-12] = 1.0 / (d[d > 1e-12] ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    # grad_i = sum_j (x_j - x_i) / d_ij^2  = sum_j (x_j - x_i) * inv_d2[i,j]
    # = (sum_j x_j * inv_d2[i,j]) - x_i * sum_j inv_d2[i,j]
    s1 = inv_d2 @ x                       # (N, 3)
    s0 = inv_d2.sum(axis=1)               # (N,)
    grad = s1 - s0[:, None] * x           # (N, 3)
    iu = np.triu_indices(N, 1)
    e = -np.sum(np.log(d[iu]))
    return e, grad


def _project_tangent(x, g):
    """Project gradient g onto tangent plane at x (unit vectors)."""
    return g - (g * x).sum(axis=1)[:, None] * x


def _fibonacci_spiral(n, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        noise = rng.normal(0, jitter, size=(n, 3))
        pts = pts + noise
        pts /= np.linalg.norm(pts, axis=1)[:, None]
    return pts


def _cart_to_sph(x):
    """(N,3) unit vectors -> (N,2) (theta, phi)."""
    theta = np.arccos(np.clip(x[:, 2], -1.0, 1.0))
    phi = np.arctan2(x[:, 1], x[:, 0])
    return np.column_stack([theta, phi])


def _sph_to_cart(tp):
    """(N,2) (theta, phi) -> (N,3) unit vectors."""
    th, ph = tp[:, 0], tp[:, 1]
    return np.column_stack([
        np.sin(th) * np.cos(ph),
        np.sin(th) * np.sin(ph),
        np.cos(th),
    ])


def _sph_energy_grad(tp):
    """Energy and gradient w.r.t. (theta, phi) via chain rule on Cartesian."""
    x = _sph_to_cart(tp)
    e, g_cart = _energy_grad(x)
    # dE/dtheta = grad_cart . d x/dtheta
    # dE/dphi   = grad_cart . d x/dphi
    th, ph = tp[:, 0], tp[:, 1]
    st, ct = np.sin(th), np.cos(th)
    cp, sp = np.cos(ph), np.sin(ph)
    dx_dth = np.column_stack([st * cp, st * sp, -ct])
    dx_dph = np.column_stack([-ct * sp, ct * cp, 0.0])
    g_th = (g_cart * dx_dth).sum(axis=1)
    g_ph = (g_cart * dx_dph).sum(axis=1)
    return e, np.column_stack([g_th, g_ph])


def _refine_cartesian(x0, max_iters=3000, tol=1e-12):
    """Tangent-plane gradient descent with backtracking line search."""
    x = x0.copy()
    e, g = _energy_grad(x)
    best_e, best_x = e, x.copy()
    step = 0.01
    for it in range(max_iters):
        if _remaining() < 15.0:
            break
        gt = _project_tangent(x, g)
        gnorm = np.linalg.norm(gt)
        if gnorm < tol:
            break
        # backtracking line search
        alpha = step
        for _ in range(30):
            x_new = x + alpha * gt
            x_new /= np.linalg.norm(x_new, axis=1)[:, None]
            e_new = _energy(x_new)
            if e_new < e:
                break
            alpha *= 0.5
        else:
            break
        # check for too-small improvement
        if e - e_new < 1e-14:
            step *= 0.5
        else:
            step = min(step * 1.05, 0.1)
        x, e = x_new, e_new
        if e < best_e:
            best_e, best_x = e, x.copy()
        g = _energy_grad(x)[1]
        gt = _project_tangent(x, g)
    return best_x, best_e


def construct():
    t0 = time.time()
    # 1) Start from a jittered Fibonacci spiral.
    x = _fibonacci_spiral(N, jitter=0.01, seed=42)
    e = _energy(x)
    best_x, best_e = x.copy(), e

    # 2) Tangent-plane gradient descent.
    x, e = _refine_cartesian(x, max_iters=5000, tol=1e-13)
    if e < best_e:
        best_x, best_e = x.copy(), e

    # 3) L-BFGS-B polish in spherical coordinates (high precision).
    if _remaining() > 20.0:
        tp = _cart_to_sph(x)
        res = minimize(
            _sph_energy_grad, tp, method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-14,
                     "maxls": 50, "maxcor": 30},
        )
        if res.success or res.fun < best_e:
            x_pol = _sph_to_cart(res.x)
            e_pol = res.fun
            if e_pol < best_e:
                best_x, best_e = x_pol, e_pol

    # 4) Final re-normalisation and validity check.
    best_x = best_x / np.linalg.norm(best_x, axis=1)[:, None]
    if not np.all(np.isfinite(best_x)):
        # Fallback to Fibonacci spiral.
        best_x = _fibonacci_spiral(N)
    # Ensure minimum pairwise distance > 1e-9.
    d = np.linalg.norm(best_x[:, None, :] - best_x[None, :, :], axis=2)
    iu = np.triu_indices(N, 1)
    if d[iu].min() < 1e-9:
        best_x = _fibonacci_spiral(N)
        best_x = best_x / np.linalg.norm(best_x, axis=1)[:, None]
    return best_x
