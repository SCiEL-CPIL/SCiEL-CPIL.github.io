"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_and_gradient(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the 3D coordinates (before projection).
    
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts)
    n = pts.shape[0]
    # Compute pairwise differences: diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # Squared distances
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist[i,j])
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j != i} (pts[i] - pts[j]) / dist[i,j]^2
    # diff[i,j] = pts[i] - pts[j]
    # grad_i = sum_j diff[i,j] / dist_sq[i,j]
    # But we need to be careful: the formula is grad_i = sum_{j != i} (pts[i] - pts[j]) / |pts[i] - pts[j]|^2
    # Note: this is the gradient of E w.r.t. pts[i] in R^3 (not yet projected to sphere)
    
    # diff[i,j] / dist_sq[i,j] summed over j
    # Set diagonal to 0
    diff_over_dist_sq = diff / dist_sq[:, :, np.newaxis]
    np.fill_diagonal(diff_over_dist_sq, 0.0)
    grad = np.sum(diff_over_dist_sq, axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _energy_on_sphere(pts):
    """Compute energy for points already on the sphere (for scoring)."""
    pts = np.asarray(pts)
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _sphere_gradient(pts):
    """
    Compute the Riemannian gradient of the energy on the sphere.
    Given the Euclidean gradient g, the Riemannian gradient is:
    grad_s = g - (g · x) x
    where x is the point on the unit sphere.
    """
    e, g = _energy_and_gradient(pts)
    # Project gradient to tangent space
    dot = np.sum(g * pts, axis=1, keepdims=True)
    grad_s = g - dot * pts
    return e, grad_s


def _frank_wolfe_sphere(pts, max_iter=200, tol=1e-8):
    """
    Frank-Wolfe (conditional gradient) method on the sphere.
    At each iteration:
    1. Compute Riemannian gradient at current point x.
    2. Solve linear minimization over the tangent space (or use a simpler approach).
    
    For the sphere, the Frank-Wolfe method works as follows:
    - The constraint is ||x|| = 1.
    - We can use the fact that the tangent space at x is {v : v · x = 0}.
    - The linear minimization over the tangent space of the linear approximation
      gives us a direction.
    
    A simpler and effective approach for the sphere: use gradient descent with
    adaptive step size and projection, but with a line search.
    
    Alternatively, use the "spherical Frank-Wolfe" which at each step:
    - Computes the gradient g at x.
    - Finds the point y on the sphere that minimizes <g, y>.
      This is y = -g / ||g|| (the point on the sphere most anti-aligned with g).
    - Moves from x toward y along the geodesic: x_new = cos(alpha) * x + sin(alpha) * (y - (y·x)x)/||y - (y·x)x||
    
    Let's implement this.
    """
    pts = _project_to_sphere(pts.copy())
    n = pts.shape[0]
    
    best_energy = _energy_on_sphere(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        e, grad_s = _sphere_gradient(pts)
        
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_s)
        if grad_norm < tol:
            break
        
        # Frank-Wolfe: find y on sphere minimizing <grad_s, y>
        # y = -grad_s / ||grad_s|| (element-wise per point)
        grad_norms = np.linalg.norm(grad_s, axis=1, keepdims=True)
        grad_norms = np.maximum(grad_norms, 1e-15)
        y = -grad_s / grad_norms
        
        # Geodesic from x to y
        # Project y onto tangent space at x: y_tan = y - (y·x)x
        y_dot_x = np.sum(y * pts, axis=1, keepdims=True)
        y_tan = y - y_dot_x * pts
        y_tan_norm = np.linalg.norm(y_tan, axis=1, keepdims=True)
        
        # For points where y is nearly parallel to x, skip or use small step
        valid = y_tan_norm > 1e-10
        y_tan_norm_safe = np.where(valid, y_tan_norm, 1.0)
        y_unit_tan = y_tan / y_tan_norm_safe
        
        # Adaptive step size: try a few steps
        best_step_energy = e
        best_step_pts = pts.copy()
        
        # Try geometric sequence of step sizes
        base_alpha = 2.0 / (it + 2)  # FW step size
        for alpha_mult in [1.0, 0.5, 0.25, 0.1]:
            alpha = base_alpha * alpha_mult
            # Geodesic: x_new = cos(alpha) * x + sin(alpha) * y_unit_tan
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)
            pts_new = cos_a * pts + sin_a * y_unit_tan
            # Ensure unit norm
            pts_new = _project_to_sphere(pts_new)
            
            e_new = _energy_on_sphere(pts_new)
            if e_new < best_step_energy:
                best_step_energy = e_new
                best_step_pts = pts_new
        
        if best_step_energy >= e:
            # No improvement, try smaller step
            break
        
        pts = best_step_pts
        e = best_step_energy
    
    return best_pts, best_energy


def _lbfgs_sphere(pts, max_iter=500):
    """
    Optimize on the sphere using L-BFGS-B with parameterization.
    Use the fact that we can optimize in R^{3N} and project, but use
    the Riemannian gradient for the direction.
    """
    pts = _project_to_sphere(pts.copy())
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        e, grad_s = _sphere_gradient(x)
        # For L-BFGS in R^3N, we need the gradient w.r.t. x_flat
        # Since we project, the effective gradient is the Riemannian gradient
        # but we need to account for the projection. A simpler approach:
        # use the Euclidean gradient but project x first.
        e2, g_euclidean = _energy_and_gradient(x)
        return e2, g_euclidean.flatten()
    
    x0 = pts.flatten()
    res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
    pts_opt = _project_to_sphere(res.x.reshape(-1, 3))
    e_opt = _energy_on_sphere(pts_opt)
    return pts_opt, e_opt


def _random_sphere(n, seed=None):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    start_time = time.time()
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _energy_on_sphere(pts)
    
    # Try multiple strategies and keep the best
    strategies = []
    
    # Strategy 1: L-BFGS from Fibonacci
    try:
        pts1, e1 = _lbfgs_sphere(pts, max_iter=300)
        strategies.append((e1, pts1))
    except Exception:
        pass
    
    # Strategy 2: Frank-Wolfe from Fibonacci
    try:
        pts2, e2 = _frank_wolfe_sphere(pts, max_iter=150)
        strategies.append((e2, pts2))
    except Exception:
        pass
    
    # Strategy 3: L-BFGS from random start
    try:
        pts_rand = _random_sphere(N, seed=42)
        pts3, e3 = _lbfgs_sphere(pts_rand, max_iter=300)
        strategies.append((e3, pts3))
    except Exception:
        pass
    
    # Strategy 4: Frank-Wolfe from random start
    try:
        pts_rand2 = _random_sphere(N, seed=123)
        pts4, e4 = _frank_wolfe_sphere(pts_rand2, max_iter=150)
        strategies.append((e4, pts4))
    except Exception:
        pass
    
    # Strategy 5: L-BFGS from another random start
    try:
        pts_rand3 = _random_sphere(N, seed=999)
        pts5, e5 = _lbfgs_sphere(pts_rand3, max_iter=300)
        strategies.append((e5, pts5))
    except Exception:
        pass
    
    # Sort by energy and pick the best
    if strategies:
        strategies.sort(key=lambda x: x[0])
        best_energy, best_pts = strategies[0]
    
    # If we still have time, do a final polish with L-BFGS on the best point
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        try:
            pts_final, e_final = _lbfgs_sphere(best_pts, max_iter=200)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = pts_final
        except Exception:
            pass
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        try:
            pts_final2, e_final2 = _frank_wolfe_sphere(best_pts, max_iter=100)
            if e_final2 < best_energy:
                best_energy = e_final2
                best_pts = pts_final2
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    # Check for near-duplicate points
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / np.maximum(norms, 1e-15)[:, np.newaxis]
    
    return best_pts

# EVOLVE-BLOCK-END
