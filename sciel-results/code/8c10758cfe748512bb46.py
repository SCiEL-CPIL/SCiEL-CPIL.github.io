"""Minimal logarithmic energy of 150 points on the sphere via Riemannian mirror-descent."""
import time
import math
import numpy as np

N = 150

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """
    Compute the logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) and its gradient
    on the ambient R^3. The gradient on the sphere is the projection of the ambient
    gradient.
    
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Pairwise distances
    # pts: (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    E = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: diff[i,j] = x_i - x_j
    # grad[i] = sum_j (x_i - x_j) / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return E, grad


def _riemannian_gradient(pts, ambient_grad):
    """Project ambient gradient onto the tangent space of the sphere at each point."""
    # Tangent space at x: {v : v . x = 0}
    # Projection: grad_tan = grad - (grad . x) x
    dots = np.sum(pts * ambient_grad, axis=1)  # (N,)
    return ambient_grad - dots[:, np.newaxis] * pts


def _exp_map(pts, v, t):
    """
    Exponential map on the unit sphere: given point x and tangent vector v,
    move by t steps along the geodesic.
    
    If |v| = 0, return x.
    Otherwise, let theta = t * |v|, and the new point is:
    x_new = cos(theta) * x + sin(theta) * (v / |v|)
    
    But we need to be careful: v is already a tangent vector (v . x = 0).
    The geodesic from x in direction v is:
    gamma(s) = cos(s * |v|) * x + sin(s * |v|) * (v / |v|)
    
    For step size t, we use s = t, so:
    x_new = cos(t * |v|) * x + sin(t * |v|) * (v / |v|)
    """
    v_norm = np.linalg.norm(v, axis=1)  # (N,)
    x_new = pts.copy()
    mask = v_norm > 1e-15
    if np.any(mask):
        theta = t * v_norm[mask]
        v_unit = v[mask] / v_norm[mask, np.newaxis]
        x_new[mask] = np.cos(theta)[:, np.newaxis] * pts[mask] + np.sin(theta)[:, np.newaxis] * v_unit
    return x_new


def _line_search(pts, grad_tan, max_iter=50, tol=1e-10):
    """
    Backtracking line search on the sphere.
    We minimize f(x) where x = exp_map(pts, grad_tan, t) for decreasing t.
    """
    f0, _ = _energy_and_grad(pts)
    t = 1.0
    for _ in range(max_iter):
        x_new = _exp_map(pts, -grad_tan, t)  # descent direction: -grad_tan
        f_new, _ = _energy_and_grad(x_new)
        if f_new < f0 - 1e-4 * t * np.sum(grad_tan * grad_tan):
            return x_new, f_new, t
        t *= 0.5
        if t < tol:
            break
    return pts, f0, t


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Start from Fibonacci spiral
    pts = _fibonacci_sphere(N)
    f, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_f = f
    
    # Riemannian mirror-descent with adaptive step sizes
    t = 1.0  # initial step size
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        _, ambient_grad = _energy_and_grad(pts)
        grad_tan = _riemannian_gradient(pts, ambient_grad)
        
        # Line search
        pts_new, f_new, t_used = _line_search(pts, grad_tan, max_iter=30, tol=1e-12)
        
        if f_new < best_f:
            best_f = f_new
            best_pts = pts_new.copy()
        
        pts = pts_new
        f = f_new
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-12:
            break
        
        # Adapt step size
        if f_new < f:
            t = min(t * 1.2, 5.0)
        else:
            t = max(t * 0.8, 0.01)
    
    # Final polish: use L-BFGS-B on the sphere via parameterization
    if time.time() < deadline - 5.0:
        try:
            from scipy.optimize import minimize
            
            # Use spherical coordinates or just do a few more Riemannian steps with smaller tolerance
            # Instead, do a final high-precision polish
            t = 0.1
            for _ in range(200):
                if time.time() > deadline:
                    break
                _, ambient_grad = _energy_and_grad(best_pts)
                grad_tan = _riemannian_gradient(best_pts, ambient_grad)
                grad_norm = np.linalg.norm(grad_tan)
                if grad_norm < 1e-14:
                    break
                
                # Small step with line search
                pts_new, f_new, _ = _line_search(best_pts, grad_tan, max_iter=20, tol=1e-14)
                if f_new < best_f:
                    best_f = f_new
                    best_pts = pts_new.copy()
        except Exception:
            pass
    
    # Normalize to ensure points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms > 1e-15
    best_pts[mask] = best_pts[mask] / norms[mask, np.newaxis]
    
    return best_pts
# EVOLVE-BLOCK-END
