"""Improved packing for 200 equal circles in a unit square using Hex Grid Init and Multi-start Simulated Annealing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def _score(pts):
    """Compute the radius for a given set of points."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    
    # Compute min distance between any two points
    dists = pdist(pts)
    min_pair_dist = np.min(dists)
    
    # Compute min distance to boundary
    dists_to_boundary = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_boundary_dist = np.min(dists_to_boundary)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r


def _hex_grid_init(seed=42):
    """Create a hexagonal grid initialization."""
    rng = np.random.default_rng(seed)
    # Estimate spacing
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    points = []
    y_step = s * np.sqrt(3) / 2.0
    x_step = s
    
    y = -0.5
    row = 0
    while y <= 0.5 + y_step:
        x = -0.5
        offset = (x_step / 2.0) * (row % 2)
        while x <= 0.5 + x_step:
            px = x + offset
            py = y
            if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                points.append([px, py])
            x += x_step
        y += y_step
        row += 1
    
    pts = np.array(points)
    
    if len(pts) == 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N].copy()

    if len(pts) > N:
        indices = rng.choice(len(pts), N, replace=False)
        return pts[indices].copy()
    elif len(pts) < N:
        extra = N - len(pts)
        random_pts = rng.uniform(-0.49, 0.49, size=(extra, 2))
        return np.vstack([pts, random_pts])
    else:
        return pts.copy()


def _run_sa(pts_init, start_time, deadline, seed=0):
    """Run Simulated Annealing for the remaining time."""
    rng = np.random.default_rng(seed)
    pts = pts_init.copy()
    current_score = _score(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    T = 0.005
    T_min = 0.00001
    alpha = 0.9999
    step_scale = 2.0 
    
    while time.time() < deadline:
        idx = rng.integers(N)
        
        step_size = T * step_scale
        move = rng.normal(0, step_size, size=2)
        
        new_pts = pts.copy()
        new_pts[idx] += move
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_score = _score(new_pts)
        
        if new_score > current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                if delta / T > -50:
                    prob = np.exp(delta / T)
                else:
                    prob = 0.0
            else:
                prob = 1.0
            
            if rng.random() < prob:
                pts = new_pts
                current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    global_best_pts = None
    global_best_score = 0.0
    
    # Multi-start strategy: 3 restarts
    # We divide the time budget into 3 chunks
    # To ensure we don't exceed deadline, we calculate sub-deadlines
    # But SA loop checks time every iteration.
    # We can just run SA 3 times, each with a fraction of the time.
    
    num_starts = 3
    time_per_start = (deadline - start_time) / num_starts
    
    for i in range(num_starts):
        current_deadline = start_time + time_per_start * (i + 1)
        # Ensure we don't go over the main deadline
        current_deadline = min(current_deadline, deadline)
        
        # Add a small buffer to the sub-deadline to avoid running over
        sub_deadline = current_deadline - 1.0 
        
        if time.time() > sub_deadline:
            break
            
        init_seed = 42 + i
        pts_init = _hex_grid_init(seed=init_seed)
        
        # Run SA
        best_pts, best_score = _run_sa(pts_init, start_time, sub_deadline, seed=100+i)
        
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_pts = best_pts
            
    if global_best_pts is None:
        # Fallback
        return _hex_grid_init(seed=42)
        
    return global_best_pts
# EVOLVE-BLOCK-END
