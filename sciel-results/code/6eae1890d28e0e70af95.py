"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """Compute the maximum common radius for given sphere centers."""
    # Distance between all pairs
    n = len(pts)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
    # Distance to walls
    min_wall = np.inf
    for i in range(n):
        for k in range(3):
            d = 0.5 - abs(pts[i][k])
            if d < min_wall:
                min_wall = d
    return min(min_dist / 2.0, min_wall)


def _local_optimize(pts, time_limit):
    """Local optimization using coordinate descent to maximize min radius."""
    start_time = time.time()
    n = len(pts)
    pts = pts.copy()
    
    # Compute initial radius
    best_r = _compute_radius(pts)
    
    # Coordinate descent: try moving each point in each direction
    step = 0.01
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 0.1:
                break
            for dim in range(3):
                for direction in [1, -1]:
                    trial = pts.copy()
                    trial[i][dim] += direction * step
                    # Check bounds
                    if abs(trial[i][dim]) > 0.5:
                        continue
                    r = _compute_radius(trial)
                    if r > best_r + 1e-12:
                        pts = trial
                        best_r = r
                        improved = True
                        no_improve_count = 0
                        # Try to take a bigger step
                        step_new = step * 1.5
                        trial2 = pts.copy()
                        trial2[i][dim] += direction * step_new
                        if abs(trial2[i][dim]) <= 0.5:
                            r2 = _compute_radius(trial2)
                            if r2 > best_r + 1e-12:
                                pts = trial2
                                best_r = r2
                                step = step_new
                            else:
                                step = step * 0.8
                        else:
                            step = step * 0.8
                    else:
                        step = max(step * 0.7, 0.001)
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 5:
                break
    
    return pts, best_r


def construct():
    # Start with cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    # Local optimization with deadline guard
    start_time = time.time()
    time_limit = 120.0  # seconds
    remaining = time_limit - (time.time() - start_time)
    
    if remaining > 5.0:
        pts, r = _local_optimize(pts, remaining - 2.0)
    
    # Final validation
    r_final = _compute_radius(pts)
    
    # Ensure all points are strictly inside the cube
    for i in range(len(pts)):
        for k in range(3):
            pts[i][k] = np.clip(pts[i][k], -0.4999999, 0.4999999)
    
    return pts
# EVOLVE-BLOCK-END
