"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(points):
    """Compute the packing radius for given centers."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside unit circle
    radii = np.hypot(points[:, 0], points[:, 1])
    if np.any(radii > 1.0):
        return 0.0
    # Compute minimum distance between all pairs
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])
            if d < min_dist:
                min_dist = d
    r = min(min_dist / 2.0, 1.0 - np.max(radii))
    return r


def _initial_hexagonal(n_points):
    """Generate initial hexagonal lattice packing."""
    # Start with a reasonable spacing
    spacing = 0.25
    points = []
    # Generate hexagonal lattice points
    max_coord = 2.0
    k = int(max_coord / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n_points, 2))
    
    # Sort by distance from center and take first n_points
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii, kind="stable")[:n_points]
    return points[idx]


def _simulated_annealing(points, time_limit):
    """Run simulated annealing to improve the packing."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    current_points = points.copy()
    current_score = best_score
    
    # Temperature schedule
    T0 = 0.05
    T_min = 0.0001
    cooling_rate = 0.995
    
    T = T0
    step_size = 0.05
    
    while T > T_min:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            break
        
        # Generate a perturbation
        idx = np.random.randint(len(current_points))
        delta = np.random.randn(2) * step_size
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Clamp to unit circle
        r = np.hypot(new_points[idx, 0], new_points[idx, 1])
        if r > 1.0:
            new_points[idx] *= (1.0 / r)
        
        new_score = _score(new_points)
        
        # Accept or reject
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if new_score > best_score:
                best_points = new_points.copy()
                best_score = new_score
        else:
            delta_s = new_score - current_score
            if np.random.random() < np.exp(delta_s / T):
                current_points = new_points
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        step_size *= 0.999
    
    return best_points, best_score


def construct():
    """Construct 50 circle centers in the unit disk maximizing the minimum radius."""
    start_time = time.time()
    time_limit = 120  # seconds, with safety margin
    
    # Generate initial packing
    points = _initial_hexagonal(N)
    
    # Run multiple restarts with simulated annealing
    best_points = points.copy()
    best_score = _score(best_points)
    
    n_restarts = 10
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
        
        # For restarts > 0, add some randomness to initial configuration
        if restart == 0:
            current_points = points.copy()
        else:
            current_points = points.copy()
            # Add random perturbation
            perturbation = np.random.randn(N, 2) * 0.1
            current_points += perturbation
            # Clamp to unit circle
            radii = np.hypot(current_points[:, 0], current_points[:, 1])
            mask = radii > 1.0
            current_points[mask] *= (1.0 / radii[mask])[:, None]
        
        # Run simulated annealing with time budget
        remaining_time = time_limit - 10 - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        sa_points, sa_score = _simulated_annealing(current_points, remaining_time)
        
        if sa_score > best_score:
            best_points = sa_points
            best_score = sa_score
    
    return best_points
# EVOLVE-BLOCK-END
