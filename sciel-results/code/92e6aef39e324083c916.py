"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE = 150.0  # wall-clock seconds; keep a safety margin
_MARGIN = 5.0      # seconds to leave before the hard kill

# EVOLVE-BLOCK-START
def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_sphere(n):
    """Generate points on unit sphere using Halton sequence in 3D."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        # Map to sphere via inverse CDF of uniform on sphere
        # Use z-coordinate from uniform, then azimuth
        z_val = 2.0 * z - 1.0  # in [-1, 1]
        theta = 2.0 * np.pi * x
        r = np.sqrt(1.0 - z_val * z_val)
        points.append([r * np.cos(theta), r * np.sin(theta), z_val])
    return np.array(points)


def _energy_and_grad(x):
    """Logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log d_ij.
    Gradient w.r.t. x_i (before projection) = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2."""
    x = np.asarray(x)
    n = x.shape[0]
    # pairwise distances
    diff = x[:, None, :] - x[None, :, :]          # (n,n,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)     # (n,n)
    np.fill_diagonal(d2, 1.0)  # avoid div-by-zero on diagonal
    d = np.sqrt(d2)
    # energy: -0.5 * sum_{i!=j} log d_ij  (each pair counted twice)
    # mask diagonal
    mask = ~np.eye(n, dtype=bool)
    energy = -0.5 * np.sum(np.log(d[mask]))
    # gradient: for each i, sum_{j!=i} (x_i - x_j)/d_ij^2
    # weight matrix w_ij = 1/d_ij^2 for i!=j
    w = np.zeros_like(d2)
    w[mask] = 1.0 / d2[mask]
    # grad_i = sum_j w_ij * (x_i - x_j) = x_i * sum_j w_ij - sum_j w_ij * x_j
    row_sum = w.sum(axis=1)
    grad = x * row_sum[:, None] - (w @ x)
    return energy, grad


def _project(x):
    """Project each row to the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return x / norms


def _riemannian_cg(x0, time_budget, max_iter=2000):
    """Riemannian conjugate gradient on the sphere.
    Uses exponential map for updates and Riemannian gradient."""
    x = _project(x0.copy())
    f, g = _energy_and_grad(x)
    best_x = x.copy()
    best_e = f

    # Riemannian gradient: project Euclidean gradient to tangent space
    def riemannian_grad(x, g_eucl):
        # Tangent projection: g_t = g_eucl - <g_eucl, x> * x
        inner = np.sum(g_eucl * x, axis=1, keepdims=True)
        return g_eucl - inner * x

    # Exponential map: exp_x(v) = x*cos(||v||) + v*sin(||v||)/||v||
    def exp_map(x, v):
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        sin_norms = np.sin(norms) / norms
        cos_norms = np.cos(norms)
        return x * cos_norms + v * sin_norms

    # Logarithmic map: log_x(y) = atan2(||y - <y,x>x||, <y,x>) * (y - <y,x>x)/||y - <y,x>x||
    def log_map(x, y):
        inner = np.sum(x * y, axis=1, keepdims=True)
        inner = np.clip(inner, -1.0 + 1e-15, 1.0 - 1e-15)
        tangent = y - inner * x
        tangent_norms = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent_norms[tangent_norms < 1e-15] = 1.0
        theta = np.arctan2(tangent_norms, inner)
        return tangent * (theta / tangent_norms)

    g_r = riemannian_grad(x, g)
    d = -g_r  # initial direction
    beta = 0.0
    start_time = time.time()

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        # Line search along geodesic: x_new = exp_x(step * d)
        step = 1.0
        x_new = exp_map(x, step * d)
        f_new, g_new = _energy_and_grad(x_new)
        g_new_r = riemannian_grad(x_new, g_new)

        # Check sufficient decrease (Riemannian Armijo)
        # Approximate directional derivative: <g_r, d>
        dir_deriv = np.sum(g_r * d)
        c1 = 1e-4
        sufficient = f_new <= f + c1 * step * dir_deriv
        count = 0
        while not sufficient and count < 25:
            step *= 0.5
            x_new = exp_map(x, step * d)
            f_new, g_new = _energy_and_grad(x_new)
            g_new_r = riemannian_grad(x_new, g_new)
            dir_deriv = np.sum(g_r * d)
            sufficient = f_new <= f + c1 * step * dir_deriv
            count += 1

        if not sufficient:
            step *= 0.1
            x_new = exp_map(x, step * d)
            f_new, g_new = _energy_and_grad(x_new)
            g_new_r = riemannian_grad(x_new, g_new)

        # Update
        x = x_new
        f = f_new
        g = g_new
        g_r = g_new_r

        if f < best_e:
            best_e = f
            best_x = x.copy()

        # Check convergence
        if np.linalg.norm(g_r) < 1e-9:
            break

        # Compute beta (Polak-Ribiere)
        g_diff = g_r - riemannian_grad(x, g)  # This is approximate
        # Actually, we need the previous Riemannian gradient
        # Let's use a simpler approach: restart if needed
        if np.sum(g_r * d) > 0:
            beta = 0.0  # restart
        else:
            # Polak-Ribiere
            num = np.sum(g_r * (g_r - _riemannian_grad(x, g)))
            den = np.sum(g_r * g_r)
            beta = max(0.0, num / den) if den > 1e-15 else 0.0

        d = -g_r + beta * d

    return _project(best_x), best_e


def _lbfgs_optimize(x0, time_budget):
    """Run L-BFGS with projection. Returns (best_x, best_energy)."""
    x0 = _project(x0.copy())
    e0, _ = _energy_and_grad(x0)
    best_x = x0.copy()
    best_e = e0

    m = 10  # memory
    W = np.zeros((m, 3 * N))
    Y = np.zeros((m, 3 * N))
    idx = 0
    x = x0.flatten().copy()
    f, g = _energy_and_grad(x.reshape(N, 3))
    g = g.flatten()

    max_iter = 5000
    start_time = time.time()

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        # L-BFGS two-loop recursion to get search direction
        q = g.copy()
        r = np.zeros(3 * N)
        alphas = np.zeros(m)

        for j in range(m - 1, -1, -1):
            if np.linalg.norm(W[j]) > 0 and np.linalg.norm(Y[j]) > 0:
                alpha = np.dot(W[j], q) / np.dot(W[j], Y[j])
                alphas[j] = alpha
                q = q - alpha * Y[j]
            else:
                alphas[j] = 0.0

        # initial Hessian approximation
        gamma = 1.0
        r = gamma * q

        for j in range(m):
            if np.linalg.norm(W[j]) > 0 and np.linalg.norm(Y[j]) > 0:
                beta = np.dot(Y[j], r) / np.dot(W[j], Y[j])
                r = r + (beta - alphas[j]) * W[j]
            else:
                pass

        d = -r

        # Line search (backtracking)
        step = 1.0
        c1 = 1e-4
        x_new = x + step * d
        x_new = _project(x_new.reshape(N, 3)).flatten()
        f_new, g_new = _energy_and_grad(x_new.reshape(N, 3))
        g_new = g_new.flatten()
        sufficient = f_new <= f + c1 * step * np.dot(g, d)
        count = 0
        while not sufficient and count < 20:
            step *= 0.5
            x_new = x + step * d
            x_new = _project(x_new.reshape(N, 3)).flatten()
            f_new, g_new = _energy_and_grad(x_new.reshape(N, 3))
            g_new = g_new.flatten()
            sufficient = f_new <= f + c1 * step * np.dot(g, d)
            count += 1

        if not sufficient:
            break

        # Update L-BFGS memory
        s = x_new - x
        y = g_new - g
        sy = np.dot(s, y)
        if sy > 1e-12:
            W[idx] = s
            Y[idx] = y
            idx = (idx + 1) % m

        x = x_new
        f = f_new
        g = g_new

        if f < best_e:
            best_e = f
            best_x = _project(x.reshape(N, 3)).copy()

        if np.linalg.norm(g) < 1e-10:
            break

    return _project(best_x), best_e


def _riemannian_grad(x, g_eucl):
    """Project Euclidean gradient to tangent space of sphere."""
    inner = np.sum(g_eucl * x, axis=1, keepdims=True)
    return g_eucl - inner * x


def construct():
    t0 = time.time()
    time_budget = _DEADLINE - _MARGIN - (time.time() - t0)

    if time_budget <= 0:
        return _fibonacci_sphere(N)

    # Phase 1: Halton initialization + Riemannian CG (global dispersion + curvature-aware)
    x_halton = _halton_sphere(N)
    x_fib = _fibonacci_sphere(N)

    # Try both initializations, keep the better one
    best_x = None
    best_e = np.inf

    # Riemannian CG from Halton
    x_h, e_h = _riemannian_cg(x_halton, time_budget * 0.35)
    if e_h < best_e:
        best_e = e_h
        best_x = x_h

    elapsed = time.time() - t0
    remaining = _DEADLINE - _MARGIN - elapsed

    if remaining > 2.0:
        # Riemannian CG from Fibonacci
        x_f, e_f = _riemannian_cg(x_fib, remaining * 0.5)
        if e_f < best_e:
            best_e = e_f
            best_x = x_f

        elapsed = time.time() - t0
        remaining = _DEADLINE - _MARGIN - elapsed

    if remaining > 1.0:
        # Phase 2: L-BFGS polish on the best result
        x_l, e_l = _lbfgs_optimize(best_x, remaining)
        if e_l < best_e:
            best_e = e_l
            best_x = x_l

    # Final projection to ensure unit sphere
    best_x = _project(best_x)

    # Ensure non-zero and finite
    if not np.all(np.isfinite(best_x)) or np.any(np.linalg.norm(best_x, axis=1) < 1e-12):
        return _fibonacci_sphere(N)

    return best_x
# EVOLVE-BLOCK-END
