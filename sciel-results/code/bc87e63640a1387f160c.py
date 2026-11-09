"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the packing radius for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Boundary distance
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = 1.0 - np.max(radii)
    return min(min_pair_dist / 2.0, min_boundary)


def _hex_lattice_init(n_points):
    """Generate initial points using a hexagonal lattice pattern."""
    # Start with a reasonable spacing
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    points = np.array(rows)
    radius = np.sqrt(np.sum(points ** 2, axis=1))
    # Keep points that can fit a circle of radius spacing/2
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    # Sort by distance from center and take the first n_points
    if len(points) >= n_points:
        dists = np.sqrt(np.sum(points ** 2, axis=1))
        indices = np.argsort(dists, kind="stable")[:n_points]
        return points[indices]
    # Fallback: uniform random
    return np.random.randn(n_points, 2) * 0.5


def _slsqp_optimize(points, time_limit):
    """Optimize using SLSQP with explicit constraints."""
    start_time = time.time()
    x0 = points.flatten()
    n = len(points)
    
    def objective(x):
        p = x.reshape(-1, 2)
        # Negate the score to minimize
        return -_compute_score(p)
    
    # Constraints: pairwise distances >= 2*r (we'll use a soft constraint via penalty)
    # and boundary: |p_i| + r <= 1
    
    # Use a penalty-based approach within SLSQP
    def penalty_func(x):
        p = x.reshape(-1, 2)
        r = _compute_score(p)
        # Penalize violations
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        radii = np.sqrt(np.sum(p ** 2, axis=1))
        max_rad = np.max(radii)
        # Penalty for boundary violation
        boundary_viol = np.maximum(0, max_rad + r - 1.0)
        # Penalty for pair violation
        pair_viol = np.maximum(0, 2.0 * r - min_pair)
        return 1000.0 * (boundary_viol ** 2 + pair_viol ** 2)
    
    def objective_with_penalty(x):
        return objective(x) + penalty_func(x)
    
    # Try SLSQP with bounds
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    best_x = x0
    best_score = -objective(x0)
    
    # Multiple restarts with different initializations
    for restart in range(5):
        if time.time() - start_time > time_limit - 5:
            break
        
        if restart == 0:
            x_init = x0
        else:
            # Perturb the initial solution
            x_init = x0 + np.random.randn(2 * n) * 0.05
            # Ensure within bounds
            x_init = np.clip(x_init, -0.99, 0.99)
        
        try:
            result = minimize(
                objective_with_penalty,
                x_init,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if result.success or result.fun < -best_score:
                current_score = -result.fun
                if current_score > best_score:
                    best_score = current_score
                    best_x = result.x
        except Exception:
            continue
    
    return best_x.reshape(-1, 2)


def _local_polish(points, time_limit):
    """Local polish using gradient-free optimization."""
    start_time = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    n = len(current_points)
    
    step_size = 0.01
    iterations = 0
    
    while time.time() - start_time < time_limit - 2:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 2:
                break
            for _ in range(10):  # Try multiple steps per point
                if time.time() - start_time > time_limit - 2:
                    break
                # Try moving point i in random direction
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                new_points = current_points.copy()
                new_points[i] = current_points[i] + direction * step_size
                
                # Check if within unit circle
                if np.linalg.norm(new_points[i]) >= 1.0:
                    continue
                
                new_score = _compute_score(new_points)
                if new_score > current_score + 1e-10:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-6:
                break
        iterations += 1
    
    return current_points


def construct():
    start_time = time.time()
    time_limit = 120.0
    
    # Generate initial hexagonal lattice
    points = _hex_lattice_init(N)
    
    # Optimize with SLSQP
    remaining = time_limit - (time.time() - start_time) - 10
    if remaining > 5:
        points = _slsqp_optimize(points, remaining)
    
    # Local polish
    remaining = time_limit - (time.time() - start_time) - 3
    if remaining > 2:
        points = _local_polish(points, remaining)
    
    # Final validation
    score = _compute_score(points)
    if score <= 0 or not np.all(np.isfinite(points)):
        # Fallback to hexagonal lattice
        points = _hex_lattice_init(N)
    
    return points
# EVOLVE-BLOCK-END
