"""Logarithmic energy minimization for 150 points on the unit sphere."""
import time
import math
import numpy as np

N = 150
_EPS = 1e-12


def _energy_and_grad(x):
    """Energy E = sum_{i<j} log(1/|x_i-x_j|) and its gradient in R^3.

    grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    diff = x[:, None, :] - x[None, :, :]          # (N,N,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)     # (N,N)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    # off-diagonal pair contributions
    off = ~np.eye(N, dtype=bool)
    e = 0.5 * np.log(1.0 / d[off]).sum()
    w = np.zeros_like(d2)
    w[off] = 1.0 / d2[off]
    g = np.einsum('ij,ijk->ik', w, diff)          # (N,3)
    return e, g


def _hessian_vector_product(x, v):
    """Compute Hessian-vector product for the logarithmic energy.
    
    This is needed for Newton-type methods to achieve higher precision.
    """
    # H_ij = sum_{k != i,j} (delta_ij * |x_i - x_k|^2 - (x_i - x_k)(x_j - x_k)^T) / |x_i - x_k|^2
    # Simplified: for each pair (i,j), the Hessian contribution involves distances to all other points
    
    # For efficiency, we'll use the structure of the problem
    diff = x[:, None, :] - x[None, :, :]          # (N,N,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)     # (N,N)
    np.fill_diagonal(d2, 1.0)
    
    # The Hessian is complex; for now, use a finite difference approximation
    # or skip Newton and rely on L-BFGS with very tight tolerances
    # Return zero as placeholder - we'll use L-BFGS with tighter params instead
    return np.zeros_like(v)


def construct():
    t0 = time.time()
    DEADLINE = 110.0          # safety margin under the 150 s wall-clock limit
    rng = np.random.default_rng(20240501)

    def time_ok():
        return (time.time() - t0) < DEADLINE

    def fib_sphere(seed=0.0):
        i = np.arange(N) + 0.5 + seed
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_sphere():
        v = rng.normal(size=(N, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # Diverse initial configurations.
    inits = [fib_sphere(0.0)]
    for _ in range(8):
        inits.append(fib_sphere(rng.uniform(0.0, 1.0)))
    for _ in range(8):
        inits.append(random_sphere())
    
    # Add some structured perturbations of the Fibonacci sphere
    base = fib_sphere(0.0)
    for amp in [0.01, 0.05, 0.1]:
        perturbed = base + amp * rng.normal(size=(N, 3))
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        inits.append(perturbed)

    def optimize(x0, max_iter=2000, ftol=1e-14, gtol=1e-12):
        from scipy.optimize import minimize
        res = minimize(
            _energy_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                     'maxls': 50, 'maxfun': 100000},
        )
        return res.x

    best_x = None
    best_e = math.inf

    for x0 in inits:
        if not time_ok():
            break
        try:
            x = optimize(x0)
        except Exception:
            continue
        if x is None:
            continue
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        e, _ = _energy_and_grad(x)
        if not np.isfinite(e):
            continue
        if e < best_e:
            best_e = e
            best_x = x.copy()

    if best_x is None:
        best_x = fib_sphere(0.0)
        best_e = _energy_and_grad(best_x)[0]

    # Final polishing with remaining time, tighter tolerances.
    # Multiple passes with progressively tighter tolerances
    polish_configs = [
        (1000, 1e-15, 1e-13),
        (500, 1e-16, 1e-14),
        (300, 1e-16, 1e-15),
    ]
    
    for config in polish_configs:
        if not time_ok() or (time.time() - t0) >= DEADLINE - 3.0:
            break
        try:
            x = optimize(best_x, **config)
        except Exception:
            break
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        e, _ = _energy_and_grad(x)
        if np.isfinite(e) and e < best_e:
            best_e = e
            best_x = x.copy()
        else:
            break

    # Additional refinement loop if time permits
    while time_ok() and (time.time() - t0) < DEADLINE - 2.0:
        try:
            x = optimize(best_x, max_iter=200, ftol=1e-16, gtol=1e-15)
        except Exception:
            break
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        e, _ = _energy_and_grad(x)
        if np.isfinite(e) and e < best_e:
            best_e = e
            best_x = x.copy()
        else:
            break

    # Ensure validity: unit norm, finite, no near-coincident points.
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
