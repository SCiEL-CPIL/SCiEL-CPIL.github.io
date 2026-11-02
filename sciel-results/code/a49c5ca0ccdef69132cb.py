import numpy as np
import time
import math

N = 150

def _energy_and_grad(points):
    """
    Compute the logarithmic energy and the Euclidean gradient for points on the sphere.
    points: (N, 3) array.
    Returns: energy (float), grad (N, 3) array.
    """
    # Compute pairwise distances
    # To avoid division by zero, we can clip distances, but the problem states points closer than 1e-9 are invalid.
    # We assume distinct points.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Mask the diagonal to avoid self-interaction
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1 / dist_ij) = - sum_{i<j} log(dist_ij)
    # dist_ij = sqrt(dist_sq_ij)
    # log(dist_ij) = 0.5 * log(dist_sq_ij)
    # Energy = -0.5 * sum_{i<j} log(dist_sq_ij)
    # We only sum i < j, so we take the upper triangle.
    triu_indices = np.triu_indices(N, k=1)
    dist_sq_triu = dist_sq[triu_indices]
    
    # Clip to prevent log(0) if points are extremely close, though we expect them to be separated.
    # If dist_sq is very small, log is -inf, energy is +inf. This is a valid signal to move them apart.
    # However, for numerical stability in gradient, we might need to be careful.
    energy = -0.5 * np.sum(np.log(dist_sq_triu))
    
    # Gradient
    # E = -0.5 * sum_{i<j} log(dist_sq_ij)
    # dE/dx_i = -0.5 * sum_{j != i} (1/dist_sq_ij) * d(dist_sq_ij)/dx_i
    # dist_sq_ij = |x_i - x_j|^2
    # d(dist_sq_ij)/dx_i = 2 * (x_i - x_j)
    # dE/dx_i = -0.5 * sum_{j != i} (1/dist_sq_ij) * 2 * (x_i - x_j)
    # dE/dx_i = - sum_{j != i} (x_i - x_j) / dist_sq_ij
    # Note: The term for j=i is 0/0, so we exclude it.
    
    # Vectorized calculation:
    # We want grad[i] = - sum_{j != i} (x_i - x_j) / dist_sq_ij
    # Let's compute the sum for all i.
    # diff is (N, N, 3). diff[i, j] = x_i - x_j.
    # dist_sq is (N, N).
    # We need to divide each vector diff[i, j] by dist_sq[i, j].
    # But dist_sq[i, i] is 0. We can set the diagonal of the divisor to 1 to avoid div by zero,
    # and then zero out the diagonal of the result.
    
    inv_dist_sq = np.zeros_like(dist_sq)
    np.divide(1.0, dist_sq, out=inv_dist_sq, where=dist_sq > 1e-12)
    # For diagonal, inv_dist_sq[i, i] will be 0 because where is False.
    
    # grad_contrib[i, j] = diff[i, j] * inv_dist_sq[i, j]
    # This is (N, N, 3)
    grad_contrib = diff * inv_dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    grad = -np.sum(grad_contrib, axis=1)
    
    return energy, grad

def _project_and_normalize(points):
    """
    Project gradient to tangent space is done by the caller.
    This function just normalizes the points to the unit sphere.
    """
    norms = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    return points / norms[:, np.newaxis]

def _optimize_single(start_points, time_limit, max_iter=2000):
    """
    Optimize a single configuration using projected gradient descent with backtracking line search.
    """
    start_time = time.time()
    points = start_points.copy()
    points = _project_and_normalize(points)
    
    energy, grad = _energy_and_grad(points)
    best_energy = energy
    best_points = points.copy()
    
    # Initial step size
    alpha = 0.01
    max_alpha = 1.0
    min_alpha = 1e-6
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . x) x
        dot_products = np.sum(points * grad, axis=1)
        grad_tangent = grad - dot_products[:, np.newaxis] * points
        
        # Backtracking line search
        # We want to find alpha such that energy(x + alpha * grad_tangent) < energy(x) - c * alpha * ||grad_tangent||^2
        # Simple backtracking: reduce alpha until energy decreases.
        
        # Normalize the step direction? No, grad_tangent magnitude varies.
        # Let's try a step in the direction of -grad_tangent.
        direction = -grad_tangent
        
        # Compute initial step energy
        new_points = points + alpha * direction
        new_points = _project_and_normalize(new_points)
        new_energy, _ = _energy_and_grad(new_points)
        
        # If energy increased, reduce alpha
        backtracks = 0
        while new_energy > energy and backtracks < 20:
            alpha *= 0.5
            if alpha < min_alpha:
                break
            new_points = points + alpha * direction
            new_points = _project_and_normalize(new_points)
            new_energy, _ = _energy_and_grad(new_points)
            backtracks += 1
        
        if new_energy < energy:
            points = new_points
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
            # Increase alpha slightly if we didn't backtrack
            if backtracks == 0:
                alpha = min(alpha * 1.1, max_alpha)
        else:
            # If we couldn't find a step, break or try a smaller alpha
            if alpha < min_alpha:
                break
            alpha *= 0.5
    
    return best_energy, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Generate initial configurations
    # Use Fibonacci sphere as a strong baseline
    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    inits = []
    # Base Fibonacci
    inits.append(fibonacci_sphere(N))
    
    # Add some perturbed Fibonacci inits
    rng = np.random.default_rng(42)
    for _ in range(3):
        base = fibonacci_sphere(N)
        noise = rng.normal(0, 0.01, base.shape)
        perturbed = base + noise
        inits.append(_project_and_normalize(perturbed))
        
    # Add random inits (less likely to be good, but helps escape local minima)
    for _ in range(2):
        random_pts = rng.normal(0, 1, (N, 3))
        inits.append(_project_and_normalize(random_pts))

    best_energy = float('inf')
    best_points = None
    
    # Allocate time for optimization
    # We have ~110 seconds.
    # Let's spend 80% of the time on the first few high-quality inits, and 20% on polishing the best.
    # Or just iterate through inits, allocating remaining time / remaining inits to each.
    
    num_inits = len(inits)
    time_per_init = 80.0 / num_inits  # Leave 30s for polish and overhead
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0: # Reserve 30s for polish
            break
            
        current_limit = min(time_per_init, deadline - time.time() - 30.0)
        if current_limit < 1.0:
            break
            
        e, pts = _optimize_single(init, current_limit)
        if e < best_energy:
            best_energy = e
            best_points = pts
            
    # 2. Polish phase
    if best_points is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            # Use a smaller initial step size for polishing
            # We can just call _optimize_single again with a very small max_alpha or tighter criteria
            # But _optimize_single uses adaptive alpha.
            # Let's just run it again with the remaining time.
            e, pts = _optimize_single(best_points, remaining_time - 2.0, max_iter=5000)
            if e < best_energy:
                best_energy = e
                best_points = pts

    if best_points is None:
        # Fallback to Fibonacci
        best_points = fibonacci_sphere(N)
        
    # Final normalization and check
    best_points = _project_and_normalize(best_points)
    
    # Check for invalid points (too close)
    # If any pair is closer than 1e-9, it's invalid.
    # We assume the optimizer prevents this, but let's verify.
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < 1e-18: # (1e-9)^2
        # This should not happen with a good optimizer, but if it does, return Fibonacci
        best_points = fibonacci_sphere(N)
        
    return best_points
