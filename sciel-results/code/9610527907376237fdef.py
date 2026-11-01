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


def _optimize_from_start(x0, time_budget):
    """Run L-BFGS-B with a time budget (seconds). Returns (best_x, best_energy)."""
    x0 = _project(x0.copy())
    e0, _ = _energy_and_grad(x0)
    best_x = x0.copy()
    best_e = e0

    def callback(xk):
        nonlocal best_x, best_e
        e, _ = _energy_and_grad(_project(xk))
        if e < best_e:
            best_e = e
            best_x = _project(xk).copy()
        return False  # continue

    # Use a wrapper that enforces the sphere constraint via projection inside the optimizer
    # We'll use a penalty-free approach: optimize on R^3 and project after each step.
    # scipy L-BFGS-B doesn't support custom step, so we do manual iteration.

    # Manual L-BFGS with projection
    m = 10  # memory
    W = np.zeros((m, 3 * N))
    Y = np.zeros((m, 3 * N))
    idx = 0
    x = x0.flatten().copy()
    f, g = _energy_and_grad(x.reshape(N, 3))
    g = g.flatten()

    rho = 1e-8
    max_iter = 5000
    start_time = time.time()

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        # L-BFGS two-loop recursion to get search direction
        s_prev = np.zeros(3 * N)
        y_prev = np.zeros(3 * N)
        q = g.copy()
        r = np.zeros(3 * N)
        alphas = np.zeros(m)
        betas = np.zeros(m)

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
                betas[j] = beta
                r = r + (beta - alpha * np.dot(W[j], Y[j]) / np.dot(W[j], Y[j])) * W[j]
            else:
                betas[j] = 0.0

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


def construct():
    t0 = time.time()
    time_budget = _DEADLINE - _MARGIN - (time.time() - t0)

    if time_budget <= 0:
        return _fibonacci_sphere(N)

    # Start from Fibonacci spiral
    x0 = _fibonacci_sphere(N)

    # First optimization pass
    best_x, best_e = _optimize_from_start(x0, time_budget * 0.6)

    elapsed = time.time() - t0
    remaining = _DEADLINE - _MARGIN - elapsed

    if remaining > 2.0:
        # Second polish pass with tighter tolerance / more iterations
        x1, e1 = _optimize_from_start(best_x, remaining)
        if e1 < best_e:
            best_x, best_e = x1, e1

    # Final projection to ensure unit sphere
    best_x = _project(best_x)

    # Ensure non-zero and finite
    if not np.all(np.isfinite(best_x)) or np.any(np.linalg.norm(best_x, axis=1) < 1e-12):
        return _fibonacci_sphere(N)

    return best_x
# EVOLVE-BLOCK-END
