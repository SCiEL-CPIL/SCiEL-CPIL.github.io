"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    # Use broadcasting: points[:, np.newaxis, :] - points[np.newaxis, :, :]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Compute log(1/d) = -log(d)
    log_terms = -np.log(dists)
    # Sum over upper triangle only
    iu = np.triu_indices(n, k=1)
    return math.fsum(log_terms[iu[0], iu[1]])


def _gradient(points, dists):
    """Compute gradient of energy w.r.t. points.
    E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # dists is n x n matrix of distances
    # grad[i] = sum_{j != i} (x_i - x_j) / dists[i,j]^2
    # Using broadcasting:
    # diff[i,j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # inv_d2[i,j] = 1 / dists[i,j]^2, but 0 on diagonal
    inv_d2 = np.zeros_like(dists)
    np.fill_diagonal(dists, 1.0)
    inv_d2 = 1.0 / (dists * dists)
    np.fill_diagonal(inv_d2, 0.0)
    # grad[i] = sum_j diff[i,j] * inv_d2[i,j]
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _conjugate_gradient(points, max_iter=5000, tol=1e-12):
    """Minimize logarithmic energy using conjugate gradient on the sphere.
    Uses a simple line search with backtracking.
    """
    n = len(points)
    x = points.copy()
    x = _project_to_sphere(x)
    
    best_x = x.copy()
    best_e = _energy(best_x)
    
    # Initial gradient
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    grad = _gradient(x, dists)
    
    # Project gradient to tangent space
    grad_t = grad - np.sum(grad * x, axis=1, keepdims=True) * x
    
    # CG variables
    p = -grad_t.copy()
    r = grad_t.copy()
    rs = np.sum(r * r)
    
    for it in range(max_iter):
        # Compute new point: x_new = normalize(x + alpha * p)
        # Use backtracking line search
        alpha = 0.01
        x_trial = _project_to_sphere(x + alpha * p)
        
        # Compute energy and gradient at trial point
        diff_t = x_trial[:, np.newaxis, :] - x_trial[np.newaxis, :, :]
        dists_t = np.sqrt(np.sum(diff_t * diff_t, axis=2))
        e_t = _energy(x_trial)
        grad_t_new = _gradient(x_trial, dists_t)
        grad_t_new_t = grad_t_new - np.sum(grad_t_new * x_trial, axis=1, keepdims=True) * x_trial
        
        # Backtracking: decrease alpha if energy didn't decrease
        max_backtrack = 30
        for bt in range(max_backtrack):
            if e_t < best_e:
                break
            alpha *= 0.5
            if alpha < 1e-15:
                break
            x_trial = _project_to_sphere(x + alpha * p)
            diff_t = x_trial[:, np.newaxis, :] - x_trial[np.newaxis, :, :]
            dists_t = np.sqrt(np.sum(diff_t * diff_t, axis=2))
            e_t = _energy(x_trial)
            grad_t_new = _gradient(x_trial, dists_t)
            grad_t_new_t = grad_t_new - np.sum(grad_t_new * x_trial, axis=1, keepdims=True) * x_trial
        
        if e_t >= best_e:
            # No improvement, break
            break
        
        # Update best
        best_e = e_t
        best_x = x_trial.copy()
        
        # Check convergence
        if np.linalg.norm(grad_t_new_t) < tol:
            x = x_trial
            break
        
        # Update CG variables
        r_new = grad_t_new_t
        rs_new = np.sum(r_new * r_new)
        
        if rs_new < 1e-30:
            break
        
        beta = rs_new / max(rs, 1e-30)
        p = -r_new + beta * p
        r = r_new
        rs = rs_new
        x = x_trial
        
        # Safety: recompute gradient periodically to avoid drift
        if it % 100 == 0:
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            grad = _gradient(x, dists)
            grad_t = grad - np.sum(grad * x, axis=1, keepdims=True) * x
            r = grad_t
            rs = np.sum(r * r)
            if rs < 1e-30:
                break
            p = -r
    
    return best_x, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    # Start with Fibonacci spiral
    x = _fibonacci_sphere(N)
    x = _project_to_sphere(x)
    
    best_x = x.copy()
    best_e = _energy(best_x)
    
    # Run conjugate gradient
    x_cg, e_cg = _conjugate_gradient(x, max_iter=10000, tol=1e-14)
    if e_cg < best_e:
        best_x = x_cg
        best_e = e_cg
    
    elapsed = time.time() - start_time
    
    # If we have time, do a second pass with tighter tolerance
    if time.time() < deadline - 5.0:
        x2 = best_x.copy()
        x2_cg, e2 = _conjugate_gradient(x2, max_iter=10000, tol=1e-15)
        if e2 < best_e:
            best_x = x2_cg
            best_e = e2
    
    # Final projection and validation
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    min_dist = 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                # Nudge point j slightly
                best_x[j] += 1e-10 * np.random.randn(3)
                best_x = _project_to_sphere(best_x)
    
    # Final check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
