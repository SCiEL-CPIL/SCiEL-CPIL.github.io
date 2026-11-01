"""Improved logarithmic energy minimizer: Riemannian GD + L-BFGS polish + restarts."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|xi-xj|) and its gradient w.r.t. pts.
    
    pts: (N, 3) array on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    np.fill_diagonal(dist, 1.0)  # avoid log(0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d_ij = dist[iu]
    energy = -np.sum(np.log(d_ij))
    
    # Gradient: d/dx_i [ -log(|x_i - x_j|) ] = -(x_i - x_j) / |x_i - x_j|^2
    # For each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: d/dx_i log(1/d) = -d/dx_i log(d) = -(x_i - x_j)/d^2
    # But we want gradient of -log(d), which is (x_i - x_j)/d^2... let me recheck.
    # Energy term: log(1/d) = -log(d)
    # d/dx_i [-log(|x_i - x_j|)] = -(1/|x_i-x_j|) * (x_i - x_j)/|x_i - x_j| = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
    
    # Vectorized: for each i, sum over j of (x_i - x_j) / d_ij^2
    # diff[i,j] = x_i - x_j
    # We need: grad[i] = -sum_j diff[i,j] / dist[i,j]^2  (j != i)
    
    # dist^2
    dist_sq = dist ** 2
    np.fill_diagonal(dist_sq, 1.0)
    
    # weight matrix: 1/d^2, with diagonal 0
    w = 1.0 / dist_sq
    np.fill_diagonal(w, 0.0)
    
    # grad[i] = -sum_j w[i,j] * diff[i,j]
    # diff[i,j] = pts[i] - pts[j]
    # sum_j w[i,j] * (pts[i] - pts[j]) = pts[i] * sum_j w[i,j] - sum_j w[i,j] * pts[j]
    row_sum = w.sum(axis=1)  # (N,)
    weighted_pts = w @ pts  # (N, 3)
    grad = -(pts * row_sum[:, np.newaxis] - weighted_pts)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _riemannian_descent(pts, max_iter=2000, tol=1e-10, deadline=None):
    """Riemannian gradient descent with adaptive step size."""
    pts = _project_to_sphere(pts.copy())
    energy, grad = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    # Adaptive step size
    lr = 1e-3
    min_lr = 1e-8
    max_lr = 1e-1
    patience = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Riemannian gradient: project Euclidean gradient to tangent space
        # tangent component = grad - (grad . x) x
        proj = (grad * pts).sum(axis=1, keepdims=True)
        tangent_grad = grad - proj * pts
        
        # Step
        new_pts = pts - lr * tangent_grad
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if new_energy < energy:
            # Accept
            pts = new_pts
            grad = new_grad
            energy = new_energy
            patience = 0
            # Increase step size cautiously
            if lr < max_lr:
                lr = min(lr * 1.2, max_lr)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        else:
            # Reject, decrease step size
            lr = max(lr * 0.5, min_lr)
            patience += 1
            if patience > 50:
                break
        
        # Convergence check
        if np.linalg.norm(tangent_grad) < tol:
            break
    
    return best_pts, best_energy


def _lbfgs_optimize(pts, deadline=None):
    """L-BFGS optimization using scipy, with sphere constraint handled via penalty or projection."""
    # We'll use L-BFGS-B on the unconstrained coordinates with a penalty for sphere constraint,
    # or better: use the fact that we can parameterize on the sphere.
    # Simplest: use L-BFGS on the flat coordinates and project after each eval.
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def jac(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, grad = _energy_and_grad(pts)
        # Project gradient to tangent
        proj = (grad * pts).sum(axis=1, keepdims=True)
        tangent_grad = grad - proj * pts
        return tangent_grad.flatten()
    
    x0 = pts.flatten()
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000})
        if res.success or res.fun < objective(x0):
            best_pts = _project_to_sphere(res.x.reshape(N, 3))
            best_energy = res.fun
            return best_pts, best_energy
    except Exception:
        pass
    
    return pts, _energy_and_grad(pts)[0]


def _random_start(seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts, best_energy = _riemannian_descent(pts, max_iter=1000, tol=1e-10, deadline=deadline)
    
    # L-BFGS polish
    if time.time() < deadline - 5:
        polished_pts, polished_energy = _lbfgs_optimize(best_pts, deadline=deadline)
        if polished_energy < best_energy:
            best_pts, best_energy = polished_pts, polished_energy
    
    # Try a few random restarts if time permits
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        seed = 42 + restart
        rand_pts = _random_start(seed)
        # Quick Riemannian descent from random start
        rd_pts, rd_energy = _riemannian_descent(rand_pts, max_iter=500, tol=1e-8, deadline=deadline - 5)
        if rd_energy < best_energy:
            best_pts, best_energy = rd_pts, rd_energy
            # Polish
            if time.time() < deadline - 5:
                pol_pts, pol_energy = _lbfgs_optimize(best_pts, deadline=deadline)
                if pol_energy < best_energy:
                    best_pts, best_energy = pol_pts, pol_energy
    
    # Final verification
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    dists = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
