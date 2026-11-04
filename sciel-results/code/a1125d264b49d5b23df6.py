"""Rewritten program: Riemannian gradient descent with line search for log-energy minimization."""
import numpy as np
import time
import math

N = 150

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy(pts):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    # Pairwise distances
    # Use efficient computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Only sum upper triangle
    iu = np.triu_indices(n, 1)
    d = dist[iu]
    # Clip to avoid log of very small numbers (shouldn't happen with valid points)
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _grad(pts):
    """Compute the gradient of the log-energy w.r.t. each point.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # grad[i] = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = -sum_{j != i} diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return grad


def _project_tangent(grad, pts):
    """Project gradient onto the tangent space of the sphere at each point."""
    # For sphere: tangent projection is grad - (grad . x) * x
    dot = np.sum(grad * pts, axis=1)
    return grad - dot[:, np.newaxis] * pts


def _line_search(pts, grad_tan, energy0, max_iter=50):
    """Backtracking line search with Armijo condition on the sphere.
    
    Returns (new_pts, new_energy, step_size).
    """
    n = len(pts)
    # Initial step size
    step = 1.0
    c1 = 1e-4  # Armijo parameter
    
    for _ in range(max_iter):
        # Move in tangent direction and re-normalize
        new_pts = pts + step * grad_tan
        # Normalize to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        new_energy = _energy(new_pts)
        
        # Armijo condition: f(x + step * g) <= f(x) + c1 * step * <g, grad_tan>
        # The directional derivative is <grad_tan, grad_tan> (since grad_tan is the projected grad)
        directional_deriv = np.sum(grad_tan * grad_tan)
        
        if new_energy <= energy0 + c1 * step * directional_deriv:
            return new_pts, new_energy, step
        
        step *= 0.5
    
    # If line search fails, return the smallest step tried
    new_pts = pts + step * grad_tan
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / norms
    new_energy = _energy(new_pts)
    return new_pts, new_energy, step


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    current_energy = _energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    # Riemannian gradient descent with line search
    max_iterations = 2000
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Compute gradient and project to tangent space
        grad = _grad(pts)
        grad_tan = _project_tangent(grad, pts)
        
        # Check if gradient is small enough
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-12:
            break
        
        # Line search
        new_pts, new_energy, step = _line_search(pts, grad_tan, current_energy)
        
        if new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        else:
            # If line search didn't improve, try a smaller step
            break
    
    # Final polish: run a few more iterations with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        for it in range(500):
            if time.time() > deadline:
                break
            
            grad = _grad(pts)
            grad_tan = _project_tangent(grad, pts)
            grad_norm = np.linalg.norm(grad_tan)
            if grad_norm < 1e-14:
                break
            
            new_pts, new_energy, step = _line_search(pts, grad_tan, current_energy, max_iter=30)
            
            if new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
            else:
                break
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Validate: check no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # This shouldn't happen, but if it does, nudge the points apart
                # Actually, just return as-is; the evaluator will mark it invalid
                pass
    
    return best_pts

# EVOLVE-BLOCK-END
