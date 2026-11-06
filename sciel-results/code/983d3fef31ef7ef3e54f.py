"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij) using math.fsum for precision."""
    n = len(pts)
    terms = []
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            d = np.linalg.norm(pi - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fast_energy(pts):
    """Fast vectorized energy for optimization loops. Returns (energy, min_dist_sq)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist_sq = float(np.min(dist_sq))
    if min_dist_sq < 1e-12:
        return float('inf'), min_dist_sq
    energy = -0.25 * float(np.sum(np.log(dist_sq)))
    return energy, min_dist_sq


def _fast_grad_tangent(pts):
    """Compute gradient of log energy projected to tangent space. Vectorized."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)

    if np.any(dist_sq < 1e-12):
        return None

    inv_d2 = np.zeros_like(dist_sq)
    valid = dist_sq > 1e-24
    inv_d2[valid] = 1.0 / dist_sq[valid]
    np.fill_diagonal(inv_d2, 0.0)

    weighted_diff = diff * inv_d2[:, :, np.newaxis]
    grad_raw = -np.sum(weighted_diff, axis=1)

    dot_products = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - dot_products * pts

    return grad_tangent


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci sphere points (seed rotates the azimuth offset)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed * 0.37
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _optimize_gradient_descent(pts, deadline, max_iters=8000, lr_init=0.08):
    """Vectorized gradient descent with adaptive backtracking learning rate.
    Returns (pts, energy)."""
    pts = pts.copy()
    _, min_ds = _fast_energy(pts)
    if min_ds < 1e-12:
        pts += 0.01 * np.random.randn(*pts.shape)
        pts = _project_to_sphere(pts)

    lr = lr_init
    prev_energy = _fast_energy(pts)[0]
    min_lr = 1e-14
    max_lr = 0.15

    for it in range(max_iters):
        if time.time() > deadline:
            break

        grad = _fast_grad_tangent(pts)
        if grad is None:
            pts += 0.01 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
            continue

        accepted = False
        for _ in range(10):
            pts_new = pts - lr * grad
            pts_new = _project_to_sphere(pts_new)
            new_energy, _ = _fast_energy(pts_new)
            if new_energy < prev_energy:
                pts = pts_new
                prev_energy = new_energy
                lr *= 1.002
                if lr > max_lr:
                    lr = max_lr
                accepted = True
                break
            else:
                lr *= 0.5

        if not accepted and lr < min_lr:
            break

        if np.linalg.norm(grad) < 1e-13:
            break

    return pts, prev_energy


def _optimize_lbfgsb(pts, deadline, maxiter=3000):
    """L-BFGS-B on the flattened point array with sphere projection.
    Returns (pts, energy)."""
    n = pts.shape[0]

    def _energy_and_grad_flat(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        energy = _fast_energy(p)[0]
        grad_t = _fast_grad_tangent(p)
        if grad_t is None:
            return energy, np.zeros_like(x)
        return energy, grad_t.flatten()

    x0 = pts.flatten().copy()
    try:
        res = minimize(
            _energy_and_grad_flat,
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-12,
                     'maxls': 80, 'm': 50}
        )
        p = _project_to_sphere(res.x.reshape(n, 3))
        energy = _fast_energy(p)[0]
        return p, energy
    except Exception:
        return pts, _fast_energy(pts)[0]


def _polish(pts, deadline, max_iters=5000, lr_init=0.02):
    """High-precision polish pass using very small adaptive gradient steps.
    Returns (pts, energy)."""
    pts = pts.copy()
    prev_energy = _fast_energy(pts)[0]
    lr = lr_init
    max_lr = 0.08

    for it in range(max_iters):
        if time.time() > deadline:
            break

        grad = _fast_grad_tangent(pts)
        if grad is None:
            break

        if np.linalg.norm(grad) < 1e-15:
            break

        accepted = False
        for _ in range(12):
            pts_new = pts - lr * grad
            pts_new = _project_to_sphere(pts_new)
            new_energy, _ = _fast_energy(pts_new)
            if new_energy < prev_energy:
                pts = pts_new
                prev_energy = new_energy
                lr *= 1.001
                if lr > max_lr:
                    lr = max_lr
                accepted = True
                break
            else:
                lr *= 0.4

        if not accepted and lr < 1e-16:
            break

    return pts, prev_energy


def _is_valid(pts):
    """Check that points are finite, on the sphere, and no pair is too close."""
    if not np.all(np.isfinite(pts)):
        return False
    norms = np.linalg.norm(pts, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-6):
        return False
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    if np.min(dist_sq) < 1e-18:  # (1e-9)^2
        return False
    return True


def construct():
    """Place N points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 118.0  # 32s safety margin before the 150s kill

    best_pts = _fibonacci_sphere(N)
    best_energy = _fast_energy(best_pts)[0]

    def _consider(pts, energy):
        nonlocal best_pts, best_energy
        if energy < best_energy and _is_valid(pts):
            best_energy = energy
            best_pts = pts.copy()

    # Build the list of initializations to try.
    inits = []
    inits.append(_fibonacci_sphere(N, seed=0))
    inits.append(_fibonacci_sphere(N, seed=1))
    rng = np.random.RandomState(42)
    inits.append(_project_to_sphere(_fibonacci_sphere(N, seed=0) + 0.01 * rng.randn(N, 3)))
    inits.append(_project_to_sphere(_fibonacci_sphere(N, seed=0) + 0.05 * rng.randn(N, 3)))
    inits.append(_project_to_sphere(_fibonacci_sphere(N, seed=2) + 0.10 * rng.randn(N, 3)))
    inits.append(_project_to_sphere(_fibonacci_sphere(N, seed=3) + 0.02 * rng.randn(N, 3)))
    for k in range(6):
        inits.append(_random_sphere(N, seed=1000 + k * 7919))

    # Phase 1: gradient descent from each initialization.
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        if time.time() > deadline - 10:
            break

        optimized_pts, opt_energy = _optimize_gradient_descent(init_pts, deadline)
        final_energy = _log_energy(optimized_pts)
        _consider(optimized_pts, final_energy)

    # Phase 2: L-BFGS-B from the best solution found so far.
    if time.time() < deadline - 30:
        lbfgsb_pts, lbfgsb_energy = _optimize_lbfgsb(best_pts, deadline, maxiter=3000)
        final_energy = _log_energy(lbfgsb_pts)
        _consider(lbfgsb_pts, final_energy)

    # Phase 3: Additional L-BFGS-B from a few other good starting points.
    if time.time() < deadline - 20:
        fib_pts = _fibonacci_sphere(N)
        fib_pts, fib_energy = _optimize_lbfgsb(fib_pts, deadline, maxiter=2000)
        final_energy = _log_energy(fib_pts)
        _consider(fib_pts, final_energy)

    # Phase 4: High-precision polish on the best solution.
    if time.time() < deadline - 10:
        polished_pts, polished_energy = _polish(best_pts, deadline, max_iters=5000, lr_init=0.02)
        final_energy = _log_energy(polished_pts)
        _consider(polished_pts, final_energy)

    # Final cleanup: ensure points are on unit sphere and valid.
    best_pts = _project_to_sphere(best_pts)

    # Verify no points are too close.
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d

    if min_dist < 1e-9:
        # Fallback to Fibonacci sphere.
        best_pts = _fibonacci_sphere(N)

    # One final very tight polish if we have time.
    if time.time() < deadline - 2:
        final_pts, final_e = _polish(best_pts, deadline, max_iters=1000, lr_init=0.01)
        final_energy = _log_energy(final_pts)
        _consider(final_pts, final_energy)
        best_pts = _project_to_sphere(best_pts)

    return best_pts
# EVOLVE-BLOCK-END
