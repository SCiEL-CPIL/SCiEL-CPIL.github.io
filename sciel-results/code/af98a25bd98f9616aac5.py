"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = points.shape[0]
    # Compute all pairwise distances
    # Use broadcasting: points[:, None, :] - points[None, :, :]
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Only upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log(0) by clipping
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _log_energy_and_grad(points):
    """Compute log energy and gradient.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    Gradient for point i:
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    (since d/dx_i [-log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2, 
     but we want -sum log, so dE/dx_i = sum_{j!=i} (x_i-x_j)/|x_i-x_j|^2)
    
    Wait, let me re-derive:
    E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} d/dx_i [-log(|x_i - x_j|)]
    For j > i: d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2
    For j < i: d/dx_i [-log(|x_j - x_i|)] = -(x_i - x_j)/|x_i - x_j|^2 (same thing since |x_j-x_i|=|x_i-x_j|)
    
    So dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]  # diff[i,j] = x_i - x_j
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    # Gradient: grad[i] = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = -sum_{j != i} diff[i,j] / dist_sq[i,j]
    
    # Create weight matrix: W[i,j] = 1/dist_sq[i,j] for i != j, 0 for i == j
    W = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    W[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = -W[i,:] @ diff[i,:] ... no, that's not right
    # grad[i] = -sum_j W[i,j] * diff[i,j]
    # This is: grad = -(W * diff).sum(axis=1) where W is broadcast
    grad = -(W * diff).sum(axis=1)
    
    # Compute energy
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _riemannian_gradient_descent(points, max_iter=20000, tol=1e-12, time_limit=110.0):
    """Perform Riemannian gradient descent on the sphere."""
    start_time = time.time()
    points = points.copy()
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    lr = 0.1
    min_lr = 1e-8
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _log_energy_and_grad(points)
        
        if energy < best_energy - 1e-15:
            best_energy = energy
            best_points = points.copy()
        
        # Project gradient onto tangent space: grad_tan = grad - (grad·x)x
        grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        # Adaptive learning rate
        # Try the step
        new_points = _project_to_sphere(points - lr * grad_tan)
        new_energy = _log_energy(new_points)
        
        if new_energy < energy:
            points = new_points
            # Slightly increase lr if improving well
            if it % 100 == 0 and it > 100:
                lr = min(lr * 1.05, 0.5)
        else:
            # Decrease learning rate
            lr *= 0.5
            if lr < min_lr:
                break
    
    return best_points, best_energy


def _simulated_annealing(points, max_iter=5000, time_limit=110.0):
    """Simple simulated annealing with random perturbations."""
    start_time = time.time()
    points = points.copy()
    current_energy = _log_energy(points)
    best_energy = current_energy
    best_points = points.copy()
    
    T = 0.1  # Initial temperature
    cooling = 0.999
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        idx = np.random.randint(N)
        # Generate random direction
        rand_dir = np.random.randn(3)
        rand_dir /= np.linalg.norm(rand_dir)
        # Move point
        new_points = points.copy()
        step_size = 0.01 * T
        new_points[idx] = _project_to_sphere(points[idx] + step_size * rand_dir)
        
        new_energy = _log_energy(new_points)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        
        T *= cooling
    
    return best_points, best_energy


def construct():
    """Construct 150 points on the unit sphere to minimize logarithmic energy."""
    total_start = time.time()
    deadline = total_start + 115.0  # Safety margin before 120s limit
    
    # Start with Fibonacci sphere
    points = _fibonacci_sphere(N)
    current_energy = _log_energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Phase 1: Riemannian gradient descent
    points, energy = _riemannian_gradient_descent(
        points, max_iter=10000, tol=1e-13, 
        time_limit=min(80.0, deadline - time.time())
    )
    if energy < best_energy:
        best_energy = energy
        best_points = points.copy()
    
    if time.time() > deadline:
        return _project_to_sphere(best_points)
    
    # Phase 2: Multiple restarts with perturbation + GD
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb the best solution
        perturbed = best_points.copy()
        noise_scale = 0.05 * (1 - r / num_restarts)
        noise = np.random.randn(N, 3) * noise_scale
        perturbed = _project_to_sphere(perturbed + noise)
        
        # Run GD
        pts, en = _riemannian_gradient_descent(
            perturbed, max_iter=5000, tol=1e-12,
            time_limit=min(15.0, deadline - time.time())
        )
        if en < best_energy:
            best_energy = en
            best_points = pts.copy()
    
    if time.time() > deadline:
        return _project_to_sphere(best_points)
    
    # Phase 3: Final high-precision polish
    points, energy = _riemannian_gradient_descent(
        best_points, max_iter=10000, tol=1e-14,
        time_limit=min(10.0, deadline - time.time())
    )
    if energy < best_energy:
        best_energy = energy
        best_points = points.copy()
    
    return _project_to_sphere(best_points)

# EVOLVE-BLOCK-END
