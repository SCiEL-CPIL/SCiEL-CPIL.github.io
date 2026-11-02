"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its Euclidean gradient for points on sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2  (for the -log term)
    Actually d/dx_i [-log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2
    So gradient of energy = sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2 ... wait let me recheck.
    
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{j!=i} d/dx_i log(d_ij) = -sum_{j!=i} (x_i - x_j)/(d_ij * d_ij) * ... 
    Actually d/dx_i log(|x_i - x_j|) = (x_i - x_j)/|x_i - x_j|^2
    So dE/dx_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    
    But since we're on the sphere, we project the gradient.
    """
    n = len(pts)
    # Compute all pairwise distances
    # Use broadcasting: diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.sqrt(np.sum(diff**2, axis=2))  # (n, n)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of -log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for each i, grad_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
    # diff[i,j] = pts[i] - pts[j]
    # dists[i,j]^2
    inv_d2 = 1.0 / (dists**2)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_d2[i,j]
    # diff is (n,n,3), inv_d2 is (n,n)
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _riemannian_cg(pts, max_iter=200, tol=1e-10, time_limit=None):
    """Riemannian conjugate gradient on the sphere."""
    n = len(pts)
    x = pts.copy()
    
    # Initial gradient
    f, g = _log_energy_and_grad(x)
    g_tangent = g - np.sum(g * x, axis=1, keepdims=True) * x  # project to tangent space
    
    d = -g_tangent.copy()
    prev_g_tangent = g_tangent.copy()
    
    best_f = f
    best_x = x.copy()
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - time_limit > 135:
            break
        
        # Line search: minimize f(x + t*d) with projection
        # Use backtracking
        alpha = 1.0
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            break
        
        for ls_iter in range(30):
            x_new = x + alpha * d
            x_new = _project_to_sphere(x_new)
            f_new, g_new = _log_energy_and_grad(x_new)
            
            # Armijo condition
            slope = np.sum(g_tangent * d)
            if f_new <= f + 1e-4 * alpha * slope:
                break
            alpha *= 0.5
        
        if f_new < best_f:
            best_f = f_new
            best_x = x_new.copy()
        
        # Check convergence
        g_new_tangent = g_new - np.sum(g_new * x_new, axis=1, keepdims=True) * x_new
        if np.linalg.norm(g_new_tangent) < tol:
            x = x_new
            break
        
        # Compute beta (Polak-Ribiere)
        g_diff = np.sum(g_new_tangent * g_tangent)
        beta = max(0, np.sum(g_new_tangent**2) / max(np.sum(g_tangent**2), 1e-30))
        
        d = -g_new_tangent + beta * d
        # Project d to tangent space at new point
        d = d - np.sum(d * x_new, axis=1, keepdims=True) * x_new
        
        x = x_new
        f = f_new
        g_tangent = g_new_tangent
    
    return best_x, best_f


def _newton_polish(pts, max_iter=50, tol=1e-12, time_limit=None):
    """Newton's method with Hessian approximation for final polish."""
    n = len(pts)
    x = pts.copy()
    
    f, g = _log_energy_and_grad(x)
    g_tangent = g - np.sum(g * x, axis=1, keepdims=True) * x
    
    best_f = f
    best_x = x.copy()
    
    # Approximate Hessian using finite differences or analytical
    # For efficiency, use a simple BFGS-like update
    # Hessian of -log|xi-xj| w.r.t. xi:
    # d2/dxi2 [-log|xi-xj|] = (3*(xi-xj)⊗(xi-xj) - d^2*I) / d^4  ... roughly
    
    # Use L-BFGS style with limited memory
    m = 10
    s_list = []
    y_list = []
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - time_limit > 138:
            break
        
        # L-BFGS two-loop recursion to compute H * g
        # For simplicity, use a basic Newton step with diagonal Hessian approximation
        # Or just use steepest descent with adaptive step
        
        # Simple approach: use the gradient direction with adaptive step size
        # This is essentially what CG does, so let's do a more careful line search
        
        d = -g_tangent
        d_norm = np.linalg.norm(d)
        if d_norm < tol:
            break
        
        alpha = 1.0
        for ls_iter in range(40):
            x_new = x + alpha * d
            x_new = _project_to_sphere(x_new)
            f_new, g_new = _log_energy_and_grad(x_new)
            
            slope = np.sum(g_tangent * d)
            if f_new <= f + 1e-5 * alpha * slope:
                break
            alpha *= 0.5
        
        if f_new < best_f:
            best_f = f_new
            best_x = x_new.copy()
        
        g_new_tangent = g_new - np.sum(g_new * x_new, axis=1, keepdims=True) * x_new
        if np.linalg.norm(g_new_tangent) < tol:
            x = x_new
            break
        
        x = x_new
        f = f_new
        g_tangent = g_new_tangent
    
    return best_x, best_f


def construct():
    # Deadline guard
    t_start = time.time()
    time_limit = t_start + 145  # Leave 5s margin before 150s kill
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit sphere
    pts = _project_to_sphere(pts)
    
    # Phase 1: Riemannian CG with moderate tolerance
    if time.time() < time_limit - 60:
        pts, f = _riemannian_cg(pts, max_iter=300, tol=1e-9, time_limit=time_limit)
    
    # Phase 2: Higher precision CG
    if time.time() < time_limit - 30:
        pts, f = _riemannian_cg(pts, max_iter=500, tol=1e-11, time_limit=time_limit)
    
    # Phase 3: Newton polish
    if time.time() < time_limit - 10:
        pts, f = _newton_polish(pts, max_iter=100, tol=1e-13, time_limit=time_limit)
    
    # Final projection
    pts = _project_to_sphere(pts)
    
    # Validate
    if not np.all(np.isfinite(pts)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts = _project_to_sphere(pts)
    
    # Check no points too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-9:
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts = _project_to_sphere(pts)
    
    return pts
# EVOLVE-BLOCK-END
