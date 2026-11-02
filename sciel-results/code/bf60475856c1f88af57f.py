"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_TIME_MARGIN = 10.0  # seconds to finish before the hard kill


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy(pts):
    """Sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)."""
    # pts: (N,3)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
    dist = np.linalg.norm(diff, axis=2)  # (N,N)
    # Mask diagonal and lower triangle
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))


def _log_energy_grad(pts):
    """Gradient of log-energy w.r.t. pts.
    E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N,N)
    dist_sq = np.maximum(dist_sq, 1e-30)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    # weight[i,j] = 1 / |x_i - x_j|^2
    weight = 1.0 / dist_sq
    np.fill_diagonal(weight, 0.0)
    # grad[i] = sum_j weight[i,j] * (x_i - x_j)
    grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    return grad


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _optimize_from_start(pts0, time_budget):
    """Run gradient-based optimization with decreasing step sizes."""
    pts = _project_to_sphere(pts0.copy())
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    # Use L-BFGS-B with custom gradient, but we need to handle sphere constraint.
    # Instead, use a simple projected gradient method with adaptive step size.
    step = 0.01
    min_step = 1e-6
    patience = 50
    no_improve = 0

    start_time = time.time()

    for iteration in range(5000):
        if time.time() - start_time > time_budget:
            break

        grad = _log_energy_grad(pts)
        # Project gradient to tangent space: remove radial component
        radial = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tangent = grad - radial * pts

        # Adaptive step size
        new_pts = _project_to_sphere(pts - step * grad_tangent)
        new_energy = _log_energy(new_pts)

        if new_energy < current_energy - 1e-12:
            pts = new_pts
            current_energy = new_energy
            no_improve = 0
            if current_energy < best_energy:
                best_pts = pts.copy()
                best_energy = current_energy
            # Increase step slightly
            step = min(step * 1.05, 0.05)
        else:
            step *= 0.5
            no_improve += 1
            if no_improve > patience or step < min_step:
                break

    return best_pts, best_energy


def _refine_with_scipy(pts0, time_budget):
    """Use scipy L-BFGS-B on a parameterization of the sphere."""
    # Use stereographic or just optimize in R^3 with projection at each step
    # Simpler: use a custom callback approach with projected gradient via scipy
    # For now, use a simple line search based approach
    
    pts = _project_to_sphere(pts0.copy())
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    start_time = time.time()
    step = 0.005
    min_step = 1e-7
    patience = 30
    no_improve = 0

    for iteration in range(3000):
        if time.time() - start_time > time_budget:
            break

        grad = _log_energy_grad(pts)
        radial = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tangent = grad - radial * pts

        # Backtracking line search
        alpha = step
        for _ in range(20):
            new_pts = _project_to_sphere(pts - alpha * grad_tangent)
            new_energy = _log_energy(new_pts)
            if new_energy < current_energy - 1e-10:
                break
            alpha *= 0.5

        if new_energy < current_energy - 1e-12:
            pts = new_pts
            current_energy = new_energy
            no_improve = 0
            if current_energy < best_energy:
                best_pts = pts.copy()
                best_energy = current_energy
        else:
            no_improve += 1
            step *= 0.8
            if no_improve > patience or step < min_step:
                break

    return best_pts, best_energy


def construct():
    global _DEADLINE
    start_time = time.time()
    deadline = start_time + _DEADLINE - _TIME_MARGIN

    # Generate multiple starting points
    starts = []
    starts.append(_fibonacci_sphere(N))

    # Add some perturbed Fibonacci starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = _fibonacci_sphere(N)
        noise = rng.normal(0, 0.1, size=pts.shape)
        pts = _project_to_sphere(pts + noise)
        starts.append(pts)

    # Add random starts
    for _ in range(3):
        pts = rng.normal(0, 1, size=(N, 3))
        pts = _project_to_sphere(pts)
        starts.append(pts)

    best_pts = None
    best_energy = float('inf')

    for idx, start_pts in enumerate(starts):
        if time.time() > deadline:
            break

        # Estimate remaining time for this start
        remaining = deadline - time.time()
        if remaining < 2:
            break

        # Allocate time: earlier starts get more time
        time_budget = min(remaining * 0.4, 20.0)

        pts, energy = _optimize_from_start(start_pts, time_budget)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

    # Final refinement pass on the best result
    if best_pts is not None and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 5:
            pts, energy = _refine_with_scipy(best_pts, min(remaining * 0.5, 15.0))
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Ensure all points are on the unit sphere and finite
    best_pts = _project_to_sphere(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)

    return best_pts


# EVOLVE-BLOCK-START
import numpy as np
import math
import time

N = 100
_DEADLINE = 150.0
_TIME_MARGIN = 10.0


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    d = np.maximum(dist[iu], 1e-15)
    return -np.sum(np.log(d))


def _grad(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    dsq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dsq, 1.0)
    dsq = np.maximum(dsq, 1e-30)
    w = 1.0 / dsq
    np.fill_diagonal(w, 0.0)
    return np.sum(w[:, :, None] * diff, axis=1)


def _proj(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(n, 1e-15)


def construct():
    t0 = time.time()
    dl = t0 + 140.0

    starts = [_fib_sphere(N)]
    rng = np.random.RandomState(123)
    for _ in range(4):
        p = _fib_sphere(N)
        p = _proj(p + rng.normal(0, 0.05, p.shape))
        starts.append(p)
    for _ in range(2):
        p = rng.normal(0, 1, (N, 3))
        starts.append(_proj(p))

    bp, be = None, float('inf')
    for s in starts:
        if time.time() > dl - 5:
            break
        p = _proj(s.copy())
        e = _energy(p)
        step = 0.01
        ns = 0
        for _ in range(3000):
            if time.time() > dl - 5:
                break
            g = _grad(p)
            rt = np.sum(p * g, axis=1, keepdims=True)
            gt = g - rt * p
            np_ = _proj(p - step * gt)
            ne = _energy(np_)
            if ne < e - 1e-12:
                p, e = np_, ne
                ns = 0
                step = min(step * 1.03, 0.05)
                if e < be:
                    be, bp = e, p.copy()
            else:
                step *= 0.5
                ns += 1
                if ns > 40 or step < 1e-6:
                    break

    if bp is None:
        bp = _fib_sphere(N)
    else:
        # final polish
        if time.time() < dl - 3:
            p = _proj(bp.copy())
            e = _energy(p)
            step = 0.003
            ns = 0
            for _ in range(2000):
                if time.time() > dl - 3:
                    break
                g = _grad(p)
                rt = np.sum(p * g, axis=1, keepdims=True)
                gt = g - rt * p
                np_ = _proj(p - step * gt)
                ne = _energy(np_)
                if ne < e - 1e-12:
                    p, e = np_, ne
                    ns = 0
                    if e < be:
                        be, bp = e, p.copy()
                else:
                    step *= 0.5
                    ns += 1
                    if ns > 30 or step < 1e-7:
                        break

    bp = _proj(bp)
    if not np.all(np.isfinite(bp)):
        bp = _fib_sphere(N)
    return bp
# EVOLVE-BLOCK-END
