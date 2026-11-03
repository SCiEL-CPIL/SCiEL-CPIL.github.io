"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Uses Simulated Annealing to maximize the minimum radius of 50 circles
    in a unit square. This approach explores the solution space by accepting
    worse solutions probabilistically to escape local minima, cooling down
    over time to converge on a high-quality packing.
    """
    rng = np.random.RandomState(42)
    
    # Initial guess: Jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].astype(float)
    # Add jitter to break symmetry
    points += rng.uniform(-0.005, 0.005, size=points.shape)
    # Ensure within bounds initially
    points = np.clip(points, -0.49, 0.49)
    
    def compute_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        
        min_wall_dist = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    best_points = points.copy()
    current_points = points.copy()
    current_r = compute_radius(current_points)
    best_r = current_r
    
    # Simulated Annealing parameters
    T_start = 0.05
    T_end = 0.0001
    n_iter = 500000
    cooling_rate = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        # Temperature
        T = T_start * (cooling_rate ** i)
        
        # Pick a random point to move
        idx = rng.randint(N)
        old_pos = current_points[idx].copy()
        
        # Perturb the position
        # Step size decreases with temperature
        step_size = T * 2.0
        delta = rng.normal(0, step_size, size=2)
        new_pos = old_pos + delta
        
        # Clamp to square boundaries
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        # Apply move
        current_points[idx] = new_pos
        
        # Calculate new radius
        new_r = compute_radius(current_points)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0 or np.random.random() < np.exp(delta_r / T):
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = current_points.copy()
        else:
            # Reject move
            current_points[idx] = old_pos
    
    return best_points
# EVOLVE-BLOCK-END
