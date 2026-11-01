"""
Optimized circle packing using force-directed relaxation with simulated annealing.
"""
import time
import numpy as np

N = 200
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_radius(pts):
    """
    Compute the largest common radius for given centres.
    Returns 0.0 if invalid.
    """
    if pts is None or pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check boundary
    # Distance to closest wall
    d_boundary = np.min(np.stack([0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])], axis=1))
    if np.min(d_boundary) < -1e-9: # Allow tiny floating point error
        return 0.0
    
    # Check pairwise distances
    # For N=200, O(N^2) is fine for a single check
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    
    r = min(np.min(d_boundary), min_pair_dist / 2.0)
    if r < 0:
        return 0.0
    return float(r)

def _initialize_hexagonal():
    """
    Initialize points in a hexagonal pattern.
    """
    # Approximate number of rows
    rows = int(np.sqrt(N * 2 / np.sqrt(3)))
    pts = []
    r = 0.5 / (rows + 1) # Approximate radius for spacing
    for i in range(rows + 2):
        y = (i - rows/2) * np.sqrt(3) * r
        offset = 0 if i % 2 == 0 else r
        # Determine how many points fit in this row
        max_x = 0.5 - r
        x_start = -0.5 + r + offset
        x = x_start
        while x <= max_x and len(pts) < N:
            pts.append([x, y])
            x += 2 * r
        # If we ran out of space in x, maybe we can squeeze in more by adjusting? 
        # For now, just take what fits.
    
    if len(pts) < N:
        # Fill remaining with random points
        remaining = N - len(pts)
        random_pts = np.random.uniform(-0.5, 0.5, size=(remaining, 2))
        pts.extend(random_pts.tolist())
        
    pts = np.array(pts[:N])
    # Add small random jitter to break symmetry
    pts += np.random.normal(0, 0.01, size=pts.shape)
    return pts

def _initialize_grid():
    """
    Initialize points in a square grid.
    """
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # If grid has more than N, take first N. If less, pad.
    if len(grid) > N:
        grid = grid[:N]
    else:
        # This shouldn't happen if m = ceil(sqrt(N)), but just in case
        while len(grid) < N:
            grid = np.vstack([grid, np.random.uniform(-0.5, 0.5, size=(1, 2))])
    return grid

def _force_directed_relaxation(pts, r_target, time_limit):
    """
    Relax the configuration using force-directed dynamics to accommodate radius r_target.
    """
    start_time = time.time()
    pts = pts.copy()
    
    # Parameters
    learning_rate = 0.05
    noise_temp = 0.05
    cooling_rate = 0.995
    
    # Neighbor grid for efficiency (simple version)
    cell_size = 2 * r_target * 1.5
    grid_dim = int(1.0 / cell_size) + 2
    
    for step in range(10000): # Max steps, but we check time
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # Using grid to find neighbors
        # Assign points to cells
        cells = {}
        for i, p in enumerate(pts):
            cx = int((p[0] + 0.5) / cell_size)
            cy = int((p[1] + 0.5) / cell_size)
            key = (cx, cy)
            if key not in cells:
                cells[key] = []
            cells[key].append(i)
        
        for i, p in enumerate(pts):
            cx = int((p[0] + 0.5) / cell_size)
            cy = int((p[1] + 0.5) / cell_size)
            
            # Check 3x3 neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key in cells:
                        for j in cells[key]:
                            if i == j:
                                continue
                            diff = pts[i] - pts[j]
                            dist = np.linalg.norm(diff)
                            if dist < 2 * r_target and dist > 1e-6:
                                # Repulsive force
                                # Spring-like force to push apart
                                # F = k * (2r - dist) * direction
                                force_mag = (2 * r_target - dist)
                                forces[i] += (diff / dist) * force_mag
                                forces[j] -= (diff / dist) * force_mag
                            
        # Boundary forces
        # If point is too close to boundary, push it in
        # Threshold: r_target
        margin = r_target
        for i in range(N):
            x, y = pts[i]
            if x < -0.5 + margin:
                forces[i, 0] += (-0.5 + margin - x) * 2.0
            elif x > 0.5 - margin:
                forces[i, 0] -= (x - (0.5 - margin)) * 2.0
            if y < -0.5 + margin:
                forces[i, 1] += (-0.5 + margin - y) * 2.0
            elif y > 0.5 - margin:
                forces[i, 1] -= (y - (0.5 - margin)) * 2.0
                
        # Update positions
        # Add noise
        noise = np.random.normal(0, noise_temp, size=pts.shape)
        pts += learning_rate * (forces + noise)
        
        # Clamp to square (hard constraint)
        pts = np.clip(pts, -0.5, 0.5)
        
        # Cool down
        noise_temp *= cooling_rate
        
    return pts

def _local_optimize(pts, time_limit):
    """
    Run force-directed relaxation with increasing radius.
    """
    start_time = time.time()
    current_r = 0.01 # Start small
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    # Try to increase radius
    max_r = 0.04 # Upper bound estimate
    while time.time() - start_time < time_limit - 2:
        # Attempt to relax at current_r
        # Use a portion of the remaining time
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1:
            break
        
        # Allocate time for this relaxation step
        step_time = min(remaining * 0.1, 5.0) # Use up to 10% of remaining time or 5s
        
        relaxed_pts = _force_directed_relaxation(pts, current_r, step_time)
        
        # Check if relaxation succeeded (radius is valid)
        achieved_r = _compute_radius(relaxed_pts)
        
        if achieved_r >= current_r - 1e-6:
            # Success, update best
            if achieved_r > best_r:
                best_r = achieved_r
                best_pts = relaxed_pts.copy()
            
            # Increase radius slightly
            current_r = achieved_r + 0.0005
            pts = relaxed_pts
        else:
            # Failure, decrease radius slightly and retry
            current_r = max(0.01, current_r * 0.95)
            pts = relaxed_pts # Keep the relaxed positions, they might be better even if r failed
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    best_pts = None
    best_r = 0.0
    
    initializers = [
        _initialize_hexagonal,
        _initialize_grid,
        lambda: np.random.uniform(-0.5, 0.5, size=(N, 2))
    ]
    
    # Run multiple restarts
    for i, init_func in enumerate(initializers):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        pts = init_func()
        remaining = DEADLINE - time.time() - 5
        if remaining <= 0:
            break
        
        # Allocate time for this restart
        # Give more time to early restarts? Or equal?
        # Let's give equal share of remaining time, but at least 5s
        allocated_time = max(5.0, remaining / (len(initializers) - i))
        
        pts, r = _local_optimize(pts, allocated_time)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        # If we have time left, do another random restart
        while time.time() - start_time < DEADLINE - 15:
            pts = np.random.uniform(-0.5, 0.5, size=(N, 2))
            # Add some structure to random restarts?
            # Maybe mix with previous best?
            # Let's just use random for now, annealing should handle it.
            
            allocated_time = max(5.0, (DEADLINE - time.time() - 5) / 2)
            pts, r = _local_optimize(pts, allocated_time)
            
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
                
            # Check if we should stop
            if time.time() - start_time > DEADLINE - 15:
                break
                
    if best_pts is None:
        best_pts = _initialize_grid()
        
    return best_pts
