"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
from scipy.spatial import cKDTree
import time

N = 200
# Safety margin: The evaluator kills at 150s. We aim to finish by 140s.
_TIME_LIMIT = 140.0

def _compute_radius(points):
    """
    Compute the maximum radius for a given set of points.
    r = min(min_pair_dist / 2, min_wall_dist)
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0
    
    n = len(points)
    if n < 2:
        min_pair = float('inf')
    else:
        tree = cKDTree(points)
        # Query 2 nearest neighbors (self + 1 other)
        dists, _ = tree.query(points, k=2)
        min_pair = np.min(dists[:, 1])
        
    # Distance to walls: min(0.5 - x, x + 0.5) for each coord
    # min_wall = min over all points of min(0.5 - |x|, 0.5 - |y|)
    # Actually, for a square [-0.5, 0.5]^2, distance to boundary is min(0.5 - |x|, 0.5 - |y|)
    wall_dists = np.min(np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1])), axis=1)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _generate_hex_grid(rows, cols, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal grid of points."""
    points = []
    # Hex grid spacing
    # For a hex grid, dx = s, dy = s * sqrt(3) / 2
    # We want to fit N points.
    # Let's estimate s.
    # Area per point approx 1.0 (unit square). s * s * sqrt(3)/2 * 2 (for staggered) ~ 1.0?
    # Actually, let's just generate a grid and scale it to fit.
    
    # Generate raw hex lattice
    raw_points = []
    for i in range(rows):
        for j in range(cols):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2.0
            raw_points.append([x, y])
            
    if not raw_points:
        return np.zeros((N, 2))
        
    raw_points = np.array(raw_points[:N])
    
    # Scale and center to fit in [-0.5, 0.5]
    x_min, y_min = raw_points[:, 0].min(), raw_points[:, 1].min()
    x_max, y_max = raw_points[:, 0].max(), raw_points[:, 1].max()
    
    # We want the bounding box of the points to be within [-0.5, 0.5]
    # Ideally, we scale such that the max extent is just under 1.0
    # But we also need to account for the radius. 
    # For now, let's scale to fit the box exactly, then local search will adjust.
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Scale factor to fit in 1.0x1.0 box
    scale = 1.0 / max(width, height)
    
    # Center
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    points = (raw_points - np.array([center_x, center_y])) * scale
    points += np.array([offset_x, offset_y])
    
    # Clip to ensure valid
    points = np.clip(points, -0.5, 0.5)
    
    return points

def _local_search(points, time_budget, seed=42):
    """
    Constraint-based local search.
    Identifies the most violated constraints (min dist pair, min wall dist)
    and moves the involved points apart or away from walls.
    """
    rng = np.random.default_rng(seed)
    current = points.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    
    # Initial step size
    step_size = 0.05
    
    while time.time() - start_time < time_budget:
        if time.time() - start_time > time_budget - 0.5:
            break
            
        tree = cKDTree(current)
        # Find nearest neighbors
        dists, idx = tree.query(current, k=2)
        nn_dists = dists[:, 1]
        nn_idxs = idx[:, 1]
        
        # Wall distances
        wall_dists = np.min(np.minimum(0.5 - np.abs(current[:, 0]), 0.5 - np.abs(current[:, 1])), axis=1)
        
        # Find the most constrained point
        # We look for the point that limits the radius the most.
        # Constraint 1: Pair distance. If d_ij < 2*r, we need to increase d_ij.
        # Constraint 2: Wall distance. If w_i < r, we need to increase w_i.
        
        # Current radius is min(min(nn_dists)/2, min(wall_dists))
        min_nn_dist = np.min(nn_dists)
        min_wall_dist = np.min(wall_dists)
        current_r_calc = min(min_nn_dist / 2.0, min_wall_dist)
        
        # If we haven't improved, or if we are stuck, try to move the most constrained points.
        
        # Identify the point with the minimum wall distance
        min_wall_idx = np.argmin(wall_dists)
        # Identify the pair with minimum distance
        min_nn_idx = np.argmin(nn_dists)
        
        # Candidate points to move
        candidates = np.unique(np.concatenate([
            np.argsort(wall_dists)[:5],
            np.argsort(nn_dists)[:5]
        ]))
        
        improved = False
        
        # Try moving each candidate in a direction that increases the local radius
        for idx in candidates:
            # Determine direction
            # 1. If wall constrained, move away from wall
            # 2. If pair constrained, move away from nearest neighbor
            
            # Check if this point is the limiting factor
            is_wall_limited = wall_dists[idx] < nn_dists[idx] / 2.0
            
            if is_wall_limited:
                # Move away from nearest wall
                x, y = current[idx]
                if x < -0.45:
                    dx, dy = 1, 0
                elif x > 0.45:
                    dx, dy = -1, 0
                elif y < -0.45:
                    dx, dy = 0, 1
                elif y > 0.45:
                    dx, dy = 0, -1
                else:
                    # Random direction if not close to wall but still limited?
                    # This shouldn't happen if is_wall_limited is true and dist > 0.45
                    angle = rng.uniform(0, 2 * np.pi)
                    dx, dy = np.cos(angle), np.sin(angle)
            else:
                # Move away from nearest neighbor
                nb = nn_idxs[idx]
                dx = current[idx, 0] - current[nb, 0]
                dy = current[idx, 1] - current[nb, 1]
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 1e-10:
                    dx /= norm
                    dy /= norm
                else:
                    angle = rng.uniform(0, 2 * np.pi)
                    dx, dy = np.cos(angle), np.sin(angle)
            
            # Try a few step sizes
            for step in [step_size, step_size * 0.5, step_size * 2.0]:
                trial = current.copy()
                trial[idx, 0] += step * dx
                trial[idx, 1] += step * dy
                
                # Clip to box
                trial[idx, 0] = np.clip(trial[idx, 0], -0.5, 0.5)
                trial[idx, 1] = np.clip(trial[idx, 1], -0.5, 0.5)
                
                trial_r = _compute_radius(trial)
                
                if trial_r > current_r + 1e-12:
                    current = trial
                    current_r = trial_r
                    improved = True
                    if trial_r > best_r:
                        best = trial.copy()
                        best_r = trial_r
                    break # If we found an improvement, stop trying other steps for this idx
            
        if not improved:
            # Reduce step size if stuck
            step_size *= 0.95
            if step_size < 1e-7:
                break
                
    return best, best_r

def _gradient_polish(points, time_budget):
    """
    Final polish using gradient ascent on a soft-min objective.
    This helps refine the position after the coarse local search.
    """
    start_time = time.time()
    current = points.copy()
    current_r = _compute_radius(current)
    
    # Use a simple projected gradient step
    # Objective: maximize r. 
    # Approximate gradient of min distance.
    
    # We will use a simple iterative scheme:
    # 1. Compute forces pushing points apart (inverse square law or linear)
    # 2. Move points along forces
    # 3. Project onto box
    
    alpha = 0.01
    beta = 0.001 # Damping
    
    while time.time() - start_time < time_budget:
        if time.time() - start_time > time_budget - 0.1:
            break
            
        tree = cKDTree(current)
        # Query a few neighbors to estimate force
        # For a smooth force, we need more than just the nearest neighbor.
        # Let's use k=5
        k = 5
        dists, idx = tree.query(current, k=k)
        
        forces = np.zeros_like(current)
        
        for i in range(N):
            for j in range(1, k):
                nb = idx[i, j]
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # Force direction: away from neighbor
                diff = current[i] - current[nb]
                # Magnitude: proportional to 1/d^2 or similar to push them apart
                # We want to maximize the minimum distance.
                # A common approach is to maximize the sum of squared distances to neighbors.
                # Gradient of sum(d_ij^2) w.r.t p_i is 2 * sum(p_i - p_j)
                forces[i] += diff / (d * d) # Inverse square force
                
        # Wall forces
        # Push away from walls if close
        wall_dists = np.min(np.minimum(0.5 - np.abs(current[:, 0]), 0.5 - np.abs(current[:, 1])), axis=1)
        # If wall dist is small, push inwards
        # Simple heuristic: if x < -0.4, push right. If x > 0.4, push left.
        for i in range(N):
            x, y = current[i]
            if x < -0.4:
                forces[i, 0] += (0.4 - abs(x)) * 10
            elif x > 0.4:
                forces[i, 0] -= (abs(x) - 0.4) * 10
            if y < -0.4:
                forces[i, 1] += (0.4 - abs(y)) * 10
            elif y > 0.4:
                forces[i, 1] -= (abs(y) - 0.4) * 10
                
        # Update
        # Normalize forces to prevent huge jumps
        force_mags = np.linalg.norm(forces, axis=1)
        max_mag = np.max(force_mags)
        if max_mag > 1e-10:
            forces /= max_mag
            
        new_pts = current + alpha * forces
        
        # Project onto box
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        
        if new_r > current_r:
            current = new_pts
            current_r = new_r
        else:
            # Reduce step size if no improvement
            alpha *= 0.9
            if alpha < 1e-6:
                break
                
    return current, current_r

def construct():
    t0 = time.time()
    
    # Multi-start strategy
    # 1. Standard Hex Grid
    # 2. Shifted Hex Grid
    # 3. Rectangular Grid
    
    candidates = []
    
    # Hex Grid
    # Estimate rows/cols for 200 points
    # Hex area per point ~ 0.866 * s^2. Total area 1.0.
    # 200 * 0.866 * s^2 = 1 => s^2 = 1/173 => s ~ 0.076
    # Width ~ cols * s. Height ~ rows * s * 0.866
    # Let's try rows=14, cols=15 (210 points, take 200)
    hex1 = _generate_hex_grid(14, 15)
    candidates.append(hex1)
    
    # Shifted Hex Grid
    hex2 = _generate_hex_grid(14, 15, offset_x=0.01, offset_y=0.01)
    candidates.append(hex2)
    
    # Rectangular Grid
    # 10x20
    rows, cols = 10, 20
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    rect = np.array([(x, y) for y in ys for x in xs])
    candidates.append(rect)
    
    # Evaluate initial scores
    best_initial = None
    best_initial_r = 0.0
    for c in candidates:
        r = _compute_radius(c)
        if r > best_initial_r:
            best_initial_r = r
            best_initial = c
    
    # Allocate time
    # Spend 20% of time on local search for the best initial
    # Spend 30% on local search for the second best (if distinct)
    # Spend 50% on polishing the global best
    
    time_per_search = (_TIME_LIMIT - (time.time() - t0)) * 0.3
    time_for_polish = (_TIME_LIMIT - (time.time() - t0)) * 0.4
    
    # Run local search on the best initial
    best_pts, best_r = _local_search(best_initial, time_per_search, seed=1)
    
    # If time remains, try another initial
    remaining = _TIME_LIMIT - (time.time() - t0)
    if remaining > 5:
        # Find the second best initial that is different
        second_best = None
        second_best_r = 0.0
        for c in candidates:
            if np.allclose(c, best_initial, atol=1e-5):
                continue
            r = _compute_radius(c)
            if r > second_best_r:
                second_best_r = r
                second_best = c
        
        if second_best is not None:
            t2, r2 = _local_search(second_best, remaining * 0.5, seed=2)
            if r2 > best_r:
                best_pts = t2
                best_r = r2
    
    # Final Polish
    remaining = _TIME_LIMIT - (time.time() - t0)
    if remaining > 2:
        best_pts, best_r = _gradient_polish(best_pts, remaining - 1)
        
    # Final check
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r
        
    return best_pts

if __name__ == "__main__":
    # Quick test
    pts = construct()
    r = _compute_radius(pts)
    print(f"Radius: {r}")
