"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 50

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _generate_hex_grid(n_points):
    """
    Generate a hexagonal lattice of points centered at origin, scaled to fit in [-0.5, 0.5].
    """
    points = []
    # Hex grid coordinates in a triangular arrangement
    # dx = 1, dy = sqrt(3)/2
    dx = 1.0
    dy = np.sqrt(3) / 2.0
    
    # Generate enough points to cover the square
    # The square is [-0.5, 0.5] x [-0.5, 0.5].
    # We'll generate points in a larger range and then filter/center.
    # Approximate number of rows needed: 0.5 / dy + 1
    num_rows = int(1.0 / dy) + 3
    
    for row in range(-num_rows, num_rows + 1):
        for col in range(-num_rows, num_rows + 1):
            x = col * dx + (0.5 if row % 2 != 0 else 0.0)
            y = row * dy
            points.append((x, y))
            
    points = np.array(points)
    
    # Center the grid
    center = np.mean(points, axis=0)
    points -= center
    
    # Scale to fit within the square with some margin
    # We want the points to be roughly within [-0.5, 0.5]
    # Let's scale based on the max absolute value
    max_val = np.max(np.abs(points))
    if max_val > 0:
        points *= (0.45 / max_val)
        
    # Sort points by distance from center to pick the most central ones
    dists_to_center = np.sum(points**2, axis=1)
    sorted_indices = np.argsort(dists_to_center)
    
    return points[sorted_indices[:n_points]]

def _sa_search(centers, rng, budget_time):
    """
    Simulated Annealing to improve packing.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Tuned parameters for N=50
    T_start = 0.05
    T_end = 1e-6
    # Estimate steps: ~1000 steps per second in numpy
    # Budget is in seconds. Let's aim for 5000 steps per second to be safe?
    # Actually, _compute_radius is O(N^2) = 2500 ops. Very fast.
    # We can do many steps.
    
    step_count = 0
    max_steps = 2000000 # Cap to prevent infinite loop if time check fails
    
    while time.time() - start_time < budget_time and step_count < max_steps:
        # Temperature schedule
        # Use a simple exponential decay based on step count relative to max_steps
        # But we don't know max_steps in advance perfectly, so use time-based?
        # Better: use a fixed number of steps for the schedule, e.g. 100,000
        schedule_steps = 100000
        progress = min(1.0, step_count / schedule_steps)
        T = T_start * (T_end / T_start) ** progress
        
        # Random perturbation
        idx = rng.integers(0, N)
        # Step size decreases as T decreases
        base_step = 0.035
        step_size = base_step * (T / T_start) ** 0.5
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius(new)
        
        if new_r >= current_r:
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            # Acceptance probability
            if T > 1e-9:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
        
        step_count += 1
        
        # Periodic restart if stuck?
        # Simple stagnation detection: if no improvement for 5000 steps, perturb more
        if step_count % 5000 == 0:
            if best_r == current_r and current_r > 0:
                 # Small random kick to all points
                 current += rng.normal(0, 0.01, current.shape)
                 current = np.clip(current, -0.5, 0.5)
                 current_r = _compute_radius(current)

    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin for 150s limit
    
    # 1. Hexagonal Initialization
    # This is structurally better than a square grid for circle packing
    best_centers = _generate_hex_grid(N)
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    best_centers = best_centers + rng.normal(0, 0.005, best_centers.shape)
    best_centers = np.clip(best_centers, -0.49, 0.49)
    
    best_radius = _compute_radius(best_centers)
    
    # 2. Main Simulated Annealing Phase
    # Use most of the budget here
    remaining = deadline - time.time()
    if remaining > 5:
        sa_budget = remaining - 5.0
        best_centers, best_radius = _sa_search(best_centers, rng, sa_budget)
        
    # 3. Final Polish: A short SA run with lower temperature to settle
    remaining = deadline - time.time()
    if remaining > 2:
        # Create a modified SA call with lower T_start
        # We can just call SA again, it will start with high T though.
        # Let's do a simple local optimization:
        # Try moving each circle slightly in the direction of increasing radius
        # This is a simple gradient-free hill climbing
        current = best_centers.copy()
        current_r = best_radius
        
        for _ in range(1000):
            if time.time() > deadline - 1.0:
                break
            improved = False
            for i in range(N):
                # Try 4 directions + center
                for dx, dy in [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001)]:
                    new = current.copy()
                    new[i, 0] += dx
                    new[i, 1] += dy
                    new = np.clip(new, -0.5, 0.5)
                    new_r = _compute_radius(new)
                    if new_r > current_r:
                        current = new
                        current_r = new_r
                        improved = True
                        break
            if not improved:
                break
                
        if current_r > best_radius:
            best_radius = current_r
            best_centers = current

    # Final check
    if not np.all(np.isfinite(best_centers)):
        # Fallback to hex grid if something went wrong
        return _generate_hex_grid(N)
        
    return best_centers
# EVOLVE-BLOCK-END
