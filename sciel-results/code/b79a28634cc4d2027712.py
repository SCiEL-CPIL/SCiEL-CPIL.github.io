"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, basinhopping


def _energy_and_grad(x_flat):
    """Compute energy and gradient for L-BFGS-B. x_flat is shape (N*3,).
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient w.r.t. x_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    """
    n = N
    points = x_flat.reshape(n, 3)
    
    # Pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # n x n x 3
    dists_sq = np.sum(diff * diff, axis=-1)  # n x n
    np.fill_diagonal(dists_sq, 1.0)  # avoid division by zero
    
    # Distances
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # = -sum_{j!=i} (x_i - x_j) / d_ij^2
    # weights[i,j] = 1/d_ij^2 for i != j
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    # grad_i = sum_j weights[i,j] * (x_j - x_i) = -x_i * sum_j weights[i,j] + sum_j weights[i,j]*x_j
    sum_weights = np.sum(weights, axis=1)  # n
    weighted_sum = weights @ points  # n x 3
    grads = -points * sum_weights[:, np.newaxis] + weighted_sum  # n x 3
    
    # Project to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    return energy, grads_tangent.flatten()


def _energy_only(x_flat):
    """Energy only (for verification)."""
    e, _ = _energy_and_grad(x_flat)
    return e


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _lbfgsb_optimize(points, maxiter=5000, tol=1e-10, deadline=None):
    """Run L-BFGS-B on the sphere. Returns (best_points, best_energy)."""
    x0 = points.flatten().copy()
    
    def objective(x):
        return _energy_only(x)
    
    def objective_and_grad(x):
        e, g = _energy_and_grad(x)
        return e, g
    
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol, 'maxls': 20}
    )
    
    best_points = _project_to_sphere(result.x.reshape(N, 3))
    best_energy = _energy_only(best_points.flatten())
    
    return best_points, best_energy


def _basin_hopping_optimize(points, n_steps=5, deadline=None):
    """Run basin hopping with L-BFGS-B as local minimizer."""
    x0 = points.flatten().copy()
    
    def objective(x):
        return _energy_only(x)
    
    def objective_and_grad(x):
        e, g = _energy_and_grad(x)
        return e, g
    
    def local_minimize(x):
        res = minimize(
            objective_and_grad, x, method='L-BFGS-B', jac=True,
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 20}
        )
        return {'x': res.x, 'fun': _energy_only(res.x)}
    
    def step(x, T):
        """Perturbation: random Gaussian step scaled by temperature."""
        rng = np.random.default_rng()
        noise = rng.standard_normal(x.shape) * (0.1 + 0.1 * T)
        x_new = x + noise
        # Renormalize each point to sphere
        pts = x_new.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        return pts.flatten()
    
    try:
        bh_result = basinhopping(
            objective,
            x0,
            niter=n_steps,
            stepsize=0.5,
            minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True, 'options': {'maxiter': 2000}},
            stepfunc=step,
            T=1.0,
            accept_test=None,
            seed=42
        )
        best_points = _project_to_sphere(bh_result.xf.reshape(N, 3))
        best_energy = _energy_only(best_points.flatten())
        return best_points, best_energy
    except Exception:
        return points, _energy_only(x0)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_points = None
    best_energy = float('inf')
    
    # Multiple diverse initializations
    inits = [
        ("fib", _fibonacci_sphere(N)),
        ("rand1", _random_uniform_sphere(N, seed=1)),
        ("rand2", _random_uniform_sphere(N, seed=42)),
        ("rand3", _random_uniform_sphere(N, seed=123)),
        ("rand4", _random_uniform_sphere(N, seed=999)),
        ("rand5", _random_uniform_sphere(N, seed=7777)),
    ]
    
    # Phase 1: L-BFGS-B from each initialization
    for name, init_pts in inits:
        if time.time() > deadline - 60:
            break
        try:
            pts, energy = _lbfgsb_optimize(init_pts, maxiter=3000, tol=1e-10)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 2: Basin hopping from the best and a couple of other good starts
    if best_points is not None and time.time() < deadline - 40:
        # Basin hopping from current best
        try:
            pts, energy = _basin_hopping_optimize(best_points, n_steps=5)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 3: More basin hopping from random restarts
    if time.time() < deadline - 30:
        for seed in [55, 200, 333]:
            if time.time() > deadline - 15:
                break
            init_pts = _random_uniform_sphere(N, seed=seed)
            try:
                pts, energy = _basin_hopping_optimize(init_pts, n_steps=3)
                if energy < best_energy:
                    best_energy = energy
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish with very tight tolerance
    if best_points is not None and time.time() < deadline - 5:
        try:
            pts, energy = _lbfgsb_optimize(best_points, maxiter=10000, tol=1e-13)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 5: One more ultra-fine polish
    if best_points is not None and time.time() < deadline - 2:
        try:
            pts, energy = _lbfgsb_optimize(best_points, maxiter=5000, tol=1e-15)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = _project_to_sphere(best_points)
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check minimum pairwise distance
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dists[iu])
    if min_dist < 1e-9:
        # Fallback: use Fibonacci
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
