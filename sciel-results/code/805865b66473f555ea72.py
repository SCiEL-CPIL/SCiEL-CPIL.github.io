"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize


def _is_valid(circles):
    """Check if circles are disjoint and inside unit square."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            if dist_sq < (min_dist - 1e-7) ** 2:
                return False
    return True


def _score(circles):
    """Compute sum of radii if valid, else 0."""
    if _is_valid(circles):
        return np.sum(circles[:, 2])
    return 0.0


def _objective(params, n_circles):
    """Objective function for optimization: negative sum of radii."""
    circles = params.reshape(n_circles, 3)
    # Penalty for invalid configurations
    penalty = 0.0
    for i in range(n_circles):
        x, y, r = circles[i]
        # Boundary penalties
        if x - r < 0:
            penalty += (x - r) ** 2 * 100
        if x + r > 1:
            penalty += (x + r - 1) ** 2 * 100
        if y - r < 0:
            penalty += (y - r) ** 2 * 100
        if y + r > 1:
            penalty += (y + r - 1) ** 2 * 100
        # Radius penalty
        if r < 0:
            penalty += r ** 2 * 100
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            min_dist = ri + rj
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2 * 100
    return -np.sum(circles[:, 2]) + penalty


def _make_initial_grid():
    """Create initial grid layout."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95  # Slightly smaller to avoid overlap
            cells.append([x, y, r])
    return np.array(cells[:N])


def _make_initial_random(seed=42):
    """Create random initial layout."""
    np.random.seed(seed)
    circles = []
    for _ in range(N):
        r = np.random.uniform(0.02, 0.1)
        x = np.random.uniform(r, 1 - r)
        y = np.random.uniform(r, 1 - r)
        circles.append([x, y, r])
    return np.array(circles)


def _optimize_from_initial(initial_circles, time_limit=100):
    """Optimize from initial configuration using Nelder-Mead."""
    n_circles = len(initial_circles)
    params = initial_circles.flatten()
    
    start_time = time.time()
    
    # Use Nelder-Mead optimization
    result = minimize(
        _objective,
        params,
        args=(n_circles,),
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return initial_circles  # Return initial if timeout
    
    optimized = result.x.reshape(n_circles, 3)
    
    # Ensure radii are non-negative
    optimized[:, 2] = np.maximum(optimized[:, 2], 0)
    
    # Clip to unit square with small margin
    for i in range(n_circles):
        r = optimized[i, 2]
        optimized[i, 0] = np.clip(optimized[i, 0], r, 1 - r)
        optimized[i, 1] = np.clip(optimized[i, 1], r, 1 - r)
    
    return optimized


def construct():
    """Main function to construct circle packing."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple initial configurations
    initial_configs = [
        _make_initial_grid(),
        _make_initial_random(seed=1),
        _make_initial_random(seed=2),
        _make_initial_random(seed=3)
    ]
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        # First check if initial config is valid
        initial_score = _score(config)
        if initial_score > best_score:
            best_score = initial_score
            best_circles = config.copy()
        
        # Try to optimize
        optimized = _optimize_from_initial(config, time_limit=30)
        opt_score = _score(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_circles = optimized.copy()
    
    # If no valid configuration found, use the grid as fallback
    if best_circles is None or not _is_valid(best_circles):
        best_circles = _make_initial_grid()
        # Make sure it's valid by adjusting radii if needed
        if not _is_valid(best_circles):
            # Scale down radii until valid
            scale = 1.0
            while not _is_valid(best_circles * scale) and scale > 0.01:
                scale *= 0.9
            best_circles = best_circles * scale
    
    return best_circles
# EVOLVE-BLOCK-END
