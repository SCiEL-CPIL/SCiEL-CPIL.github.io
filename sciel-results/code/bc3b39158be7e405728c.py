"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 150
DEADLINE = 150.0
SAFETY = 5.0  # seconds safety margin


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i (constrained to sphere) = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise distances
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    np.fill_diagonal(dist, 0.0)  # will mask out
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    e = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, dist[i,j] = |x_i - x_j|
    # grad_i = sum_j (x_i - x_j) / dist[i,j]^2  for j != i
    inv_dist2 = np.zeros_like(dist)
    mask = dist > 0
    inv_dist2[mask] = 1.0 / (dist[mask] ** 2)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist2[i,j]
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return e, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _backtracking_line_search(pts, e0, grad, alpha_max=1.0, beta=0.5, c1=1e-4, max_iter=30):
    """Backtracking line search for gradient descent on sphere.
    
    We take step in direction -grad (projected to tangent space), then re-normalize.
    """
    n = len(pts)
    # Project gradient to tangent space: grad_tangent = grad - (grad . x) * x
    grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    # Ensure we're moving downhill
    alpha = alpha_max
    for _ in range(max_iter):
        # Tentative step
        new_pts = pts - alpha * grad_tangent
        new_pts = _project_to_sphere(new_pts)
        
        e_new, _ = _energy_and_grad(new_pts)
        
        # Armijo condition
        if e_new <= e0 + c1 * alpha * np.sum(grad * grad_tangent):
            return new_pts, e_new, alpha
        
        alpha *= beta
    
    # If no acceptable step found, try a very small step
    new_pts = pts - alpha * grad_tangent
    new_pts = _project_to_sphere(new_pts)
    e_new, _ = _energy_and_grad(new_pts)
    return new_pts, e_new, alpha


def construct():
    t_start = time.time()
    
    # Initial: golden-angle Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _project_to_sphere(pts)
    
    e, grad = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_e = e
    
    # Gradient descent with backtracking line search
    alpha = 1.0
    stagnant = 0
    min_improvement = 1e-12
    
    while True:
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - SAFETY:
            break
        
        new_pts, e_new, alpha = _backtracking_line_search(pts, e, grad, alpha_max=alpha * 2.0)
        
        if e_new < best_e - min_improvement:
            best_pts = new_pts.copy()
            best_e = e_new
            stagnant = 0
        else:
            stagnant += 1
        
        pts = new_pts
        e = e_new
        grad, _ = _energy_and_grad(pts)
        # Recompute gradient properly
        _, grad = _energy_and_grad(pts)
        
        # If we've made very little progress for many iterations, try to kick out of local min
        if stagnant > 50:
            # Add small random perturbation
            noise = np.random.randn(*pts.shape) * 0.01
            perturbed = _project_to_sphere(pts + noise)
            e_pert, _ = _energy_and_grad(perturbed)
            if e_pert < e:
                pts = perturbed
                e = e_pert
                _, grad = _energy_and_grad(pts)
                stagnant = 0
            else:
                # Reduce perturbation
                noise = np.random.randn(*pts.shape) * 0.001
                perturbed = _project_to_sphere(pts + noise)
                e_pert, _ = _energy_and_grad(perturbed)
                if e_pert < e:
                    pts = perturbed
                    e = e_pert
                    _, grad = _energy_and_grad(pts)
                stagnant = 0
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-10:
            break
    
    # Final polish: more iterations with tighter tolerance
    elapsed = time.time() - t_start
    if elapsed < DEADLINE - SAFETY - 2.0:
        # Do a few more steps with smaller alpha
        alpha = 0.1
        for _ in range(200):
            if time.time() - t_start > DEADLINE - SAFETY:
                break
            new_pts, e_new, alpha = _backtracking_line_search(pts, e, grad, alpha_max=alpha * 1.5, max_iter=20)
            if e_new < best_e - 1e-14:
                best_pts = new_pts.copy()
                best_e = e_new
            pts = new_pts
            e = e_new
            _, grad = _energy_and_grad(pts)
    
    return _project_to_sphere(best_pts)
