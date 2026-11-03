"""Optimized logarithmic energy minimizer for 150 points on the sphere."""
import time
import math
import numpy as np

N = 150

def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy(pts):
    """Compute logarithmic energy using math.fsum for accuracy."""
    n = len(pts)
    # Compute all pairwise distances
    # pts has shape (n, 3)
    # diff[i,j] = pts[i] - pts[j]
    # We want sum over i<j of log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    # Compute squared distances
    # d2[i,j] = |pts[i] - pts[j]|^2
    # Use broadcasting: pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    
    # Get upper triangle indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    d2_pairs = d2[i_idx, j_idx]
    
    # Avoid log(0) - should not happen if points are distinct
    # But add small epsilon just in case
    d2_pairs = np.maximum(d2_pairs, 1e-30)
    
    # log(1/d) = -0.5 * log(d^2)
    log_terms = -0.5 * np.log(d2_pairs)
    
    # Use math.fsum for accurate summation
    return math.fsum(log_terms.tolist())

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # For each pair i<j, contribution to grad[i] and grad[j]
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j>i} (1/d_ij) * d(d_ij)/d(x_i) + ... 
    # Actually: E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j != i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j != i} (x_i - x_j)/d_ij^2
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-30:
                continue
            d = math.sqrt(d2)
            grad[i] += -diff / d2
    
    return grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_projection(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    # grad_tangent = grad - (grad . x) * x
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts

def _lbfgs(pts, max_iter=5000, tol=1e-10):
    """Simple L-BFGS optimizer for the sphere constraint."""
    n = len(pts)
    m = 10  # L-BFGS memory
    
    best_pts = pts.copy()
    best_energy = _log_energy(best_pts)
    
    # L-BFGS state
    s_list = []  # step directions
    y_list = []  # gradient differences
    g_prev = None
    
    for it in range(max_iter):
        # Compute energy and gradient
        energy = _log_energy(pts)
        
        # Check if better
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        grad = _log_energy_grad(pts)
        grad_t = _tangent_projection(pts, grad)
        
        # Check convergence
        gnorm = np.linalg.norm(grad_t)
        if gnorm < tol:
            break
        
        # L-BFGS two-loop recursion to compute search direction
        if g_prev is not None and len(s_list) > 0:
            # Two-loop recursion
            v = []
            alphas = []
            q = grad_t.copy()
            
            for k in range(len(s_list) - 1, -1, -1):
                s_k = s_list[k]
                y_k = y_list[k]
                ys = np.sum(s_k, axis=1)
                yy = np.sum(y_k, axis=1)
                alpha_k = np.sum(s_k * q, axis=1) / ys if ys > 1e-15 else 0.0
                alphas.append(alpha_k)
                q = q - alpha_k * y_k
                v.append(alpha_k)
            
            # Initial Hessian approximation
            if len(y_list) > 0:
                yy_last = np.sum(y_list[-1], axis=1)
                ys_last = np.sum(s_list[-1], axis=1)
                gamma = ys_last / yy_last if yy_last > 1e-15 else 1.0
            else:
                gamma = 1.0
            
            r = gamma * q
            
            for k in range(len(s_list)):
                s_k = s_list[k]
                y_k = y_list[k]
                beta = np.sum(y_k * r, axis=1)
                if np.sum(s_k * s_k, axis=1) > 1e-15:
                    beta /= np.sum(s_k * s_k, axis=1)
                r = r + (alphas[k] - beta) * s_k
            
            direction = -r
        else:
            direction = -grad_t
        
        # Project direction to tangent space
        dir_dot = np.sum(pts * direction, axis=1, keepdims=True)
        direction = direction - dir_dot * pts
        
        # Line search (backtracking)
        step_size = 1.0
        cur_energy = energy
        cur_grad = grad_t
        
        for ls in range(30):
            new_pts = pts + step_size * direction
            new_pts = _project_to_sphere(new_pts)
            
            new_energy = _log_energy(new_pts)
            new_grad = _log_energy_grad(new_pts)
            new_grad_t = _tangent_projection(new_pts, new_grad)
            
            # Armijo condition
            slope = np.sum(direction * cur_grad)
            if new_energy < cur_energy + 1e-4 * step_size * slope:
                break
            step_size *= 0.5
        
        # Update L-BFGS state
        s_new = new_pts - pts
        y_new = new_grad_t - cur_grad
        sy = np.sum(s_new * y_new, axis=1)
        
        if sy > 1e-15:
            s_list.append(s_new)
            y_list.append(y_new)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        g_prev = cur_grad
        pts = new_pts
    
    return best_pts, best_energy

def _gradient_descent(pts, max_iter=20000, lr_init=0.01):
    """Simple gradient descent with adaptive step size."""
    best_pts = pts.copy()
    best_energy = _log_energy(best_pts)
    
    lr = lr_init
    
    for it in range(max_iter):
        energy = _log_energy(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        grad = _log_energy_grad(pts)
        grad_t = _tangent_projection(pts, grad)
        
        gnorm = np.linalg.norm(grad_t)
        if gnorm < 1e-12:
            break
        
        # Normalize gradient for stable steps
        if gnorm > 1e-15:
            grad_norm = grad_t / gnorm
        else:
            break
        
        # Line search
        step = lr
        improved = False
        for ls in range(20):
            new_pts = pts + step * grad_norm
            new_pts = _project_to_sphere(new_pts)
            new_energy = _log_energy(new_pts)
            
            if new_energy < energy:
                improved = True
                pts = new_pts
                break
            step *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    
    # Phase 1: Gradient descent for quick improvement
    pts, e1 = _gradient_descent(pts, max_iter=5000, lr_init=0.05)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return pts
    
    # Phase 2: L-BFGS for refinement
    pts, e2 = _lbfgs(pts, max_iter=3000, tol=1e-10)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return pts
    
    # Phase 3: Fine gradient descent polish
    pts, e3 = _gradient_descent(pts, max_iter=10000, lr_init=0.001)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return pts
    
    # Final verification: ensure points are on unit sphere and valid
    pts = _project_to_sphere(pts)
    
    # Check for minimum distance
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Points too close, perturb slightly
        pts = _fibonacci_sphere(N)
    
    return pts
