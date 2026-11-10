"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate hexagonal lattice points inside unit circle."""
    best_points = None
    best_r = 0.0
    
    for spacing in np.linspace(0.28, 0.35, 30):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) >= N:
            # Sort by distance from center, take innermost N
            points = points[np.argsort(radius[mask], kind="stable")[:N]]
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
    
    return best_points, best_r

def _anneal(points, start_r, time_budget=100.0):
    """Simulated annealing to improve packing."""
    t0 = time.time()
    current = points.copy().astype(float)
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Temperature schedule
    T0 = 0.05
    T_end = 0.001
    n_steps = 50000
    T = T0
    
    for step in range(n_steps):
        if time.time() - t0 > time_budget:
            break
        
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Random perturbation
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = T * np.random.exponential(1.0)
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        old_x, old_y = current[i, 0], current[i, 1]
        new_x, new_y = old_x + dx, old_y + dy
        
        # Check if still inside unit circle (with margin)
        if np.hypot(new_x, new_y) > 1.0:
            T *= 0.999
            continue
        
        current[i, 0] = new_x
        current[i, 1] = new_y
        
        new_r = _compute_radius(current)
        
        # Accept if better or with probability exp(-(new_r - current_r)/T)
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            delta = current_r - new_r
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / max(T, 1e-10))
            if np.random.random() < accept_prob:
                current_r = new_r
            else:
                # Revert
                current[i, 0] = old_x
                current[i, 1] = old_y
        
        # Cool down
        T = T0 * (T_end / T0) ** (step / n_steps)
    
    return best, best_r

def construct():
    t0 = time.time()
    
    # Get initial hexagonal packing
    points, r = _hexagonal_packing()
    
    if points is None:
        # Fallback: simple grid
        points = np.zeros((N, 2))
        r = 0.0
    
    # Run annealing with time budget
    time_left = 120.0 - (time.time() - t0) - 10.0  # 10s safety margin
    if time_left > 5:
        points, r = _anneal(points, r, time_budget=time_left)
    
    return points
# EVOLVE-BLOCK-END
