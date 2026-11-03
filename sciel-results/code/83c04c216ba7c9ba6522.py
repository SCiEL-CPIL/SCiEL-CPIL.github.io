"""
Packing 200 equal circles in a unit square [-1/2, 1/2]^2.
Structural approach: multi-restart with hexagonal initializations,
force-directed relaxation (repulsion between circles + walls),
followed by precise local search using simulated annealing on the
exact radius objective, then coordinate-descent polishing.
"""
import numpy as np
import time
import math

N = 200
DEADLINE = time.time() + 140.0  # 140s out of 150s total, 10s safety margin

def _compute_radius(centers):
    """Compute the valid packing radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to walls
    min_wall = np.min(0.5 - np.abs(centers))
    if min_wall < 0:
        return 0.0
    # Minimum pair distance
    min_pair = np.inf
    for i in range(N):
        diffs = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if len(dists) > 0:
            min_pair = min(min_pair, np.min(dists))
    if not np.isfinite(min_pair):
        return 0.0
    r = min(min_wall, min_pair / 2.0)
    return max(0.0, r)

def _force_directed(centers, steps=200, deadline=None):
    """
    Force-directed relaxation: repel circles from each other and from walls.
    Uses a simple spring-based model.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Target distance: we want circles to be as far apart as possible
    # Use the current radius as a hint for target separation
    target_sep = current_r * 2.0 + 0.001
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion (only for close pairs to save time)
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < target_sep * 1.5:
                    if dist > 1e-10:
                        # Repulsive force
                        f = (target_sep - dist) / dist * 0.1
                        forces[i] += diff * f
                        forces[j] -= diff * f
                    else:
                        # Coincident points
                        forces[i] += np.random.randn(2) * 0.01
                        forces[j] -= np.random.randn(2) * 0.01
        
        # Wall repulsion
        for i in range(N):
            # Left wall
            if centers[i][0] < -0.5 + target_sep:
                forces[i][0] += (-0.5 + target_sep - centers[i][0]) * 0.5
            # Right wall
            if centers[i][0] > 0.5 - target_sep:
                forces[i][0] -= (centers[i][0] - (0.5 - target_sep)) * 0.5
            # Bottom wall
            if centers[i][1] < -0.5 + target_sep:
                forces[i][1] += (-0.5 + target_sep - centers[i][1]) * 0.5
            # Top wall
            if centers[i][1] > 0.5 - target_sep:
                forces[i][1] -= (centers[i][1] - (0.5 - target_sep)) * 0.5
        
        # Update positions with adaptive step size
        step_size = 0.01 * max(0.1, 1.0 - step / steps)
        centers += forces * step_size
        centers = np.clip(centers, -0.5, 0.5)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers, best_r

def _sim_annealing(centers, n_iter=5000, t_start=0.005, t_end=0.0001, deadline=None):
    """
    Simulated annealing on the exact radius objective.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    rng = np.random.RandomState(12345)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        t = t_start * (t_end / t_start) ** (it / max(1, n_iter))
        
        # Pick a random circle to move
        i = rng.randint(N)
        orig = centers[i].copy()
        
        # Random displacement, decreasing with temperature
        step = t * 0.3
        delta = rng.randn(2) * step
        new_pos = orig + delta
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        centers[i] = new_pos
        new_r = _compute_radius(centers)
        
        if new_r >= current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            delta_r = new_r - current_r
            # Accept with probability exp(delta_r / t)
            if delta_r < -50 * t:
                # Definitely reject
                centers[i] = orig
            else:
                if rng.rand() < math.exp(delta_r / max(t, 1e-10)):
                    current_r = new_r
                else:
                    centers[i] = orig
    
    return best_centers, best_r

def _coordinate_descent(centers, deadline=None):
    """
    Precise coordinate descent: try small moves in 8 directions for each point.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    directions = [
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ]
    directions = np.array(directions, dtype=float)
    directions /= np.sqrt(2)
    
    for step in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]:
        if deadline is not None and time.time() > deadline:
            break
        
        improved = True
        while improved:
            improved = False
            for i in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                
                orig = centers[i].copy()
                best_local_r = current_r
                best_local_pos = orig
                
                for d in directions:
                    new_pos = orig + d * step
                    if new_pos[0] < -0.5 or new_pos[0] > 0.5:
                        continue
                    if new_pos[1] < -0.5 or new_pos[1] > 0.5:
                        continue
                    
                    centers[i] = new_pos
                    r = _compute_radius(centers)
                    if r > best_local_r:
                        best_local_r = r
                        best_local_pos = new_pos.copy()
                
                centers[i] = best_local_pos
                if not np.allclose(best_local_pos, orig):
                    improved = True
                    if best_local_r > current_r:
                        current_r = best_local_r
                        if current_r > best_r:
                            best_r = current_r
                            best_centers = centers.copy()
    
    return best_centers, best_r

def _hex_grid(n_rows):
    """Generate a hexagonal lattice configuration for N points."""
    n_cols = math.ceil(N / n_rows)
    d = 1.0 / (n_cols - 0.5)
    row_spacing = d * math.sqrt(3) / 2.0
    
    pts = []
    for row in range(n_rows):
        y = -0.5 + d/2.0 + row * row_spacing
        if y > 0.5 - d/2.0:
            break
        if row % 2 == 0:
            n_in_row = n_cols
            x_start = -0.5 + d/2.0
        else:
            n_in_row = n_cols - 1
            x_start = -0.5 + d
        
        for col in range(n_in_row):
            x = x_start + col * d
            if x <= 0.5 - d/2.0:
                pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) >= N:
        pts = pts[:N]
    elif len(pts) > 0:
        while len(pts) < N:
            idx = np.random.randint(len(pts))
            new_pt = pts[idx] + np.random.randn(2) * d * 0.1
            new_pt = np.clip(new_pt, -0.5, 0.5)
            pts = np.vstack([pts, new_pt])
    
    return pts

def _square_grid():
    """Generate a square grid configuration for N points."""
    m = int(math.ceil(math.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def construct():
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Hexagonal grids with different row counts
    for n_rows in [12, 13, 14, 15, 16, 17, 18]:
        config = _hex_grid(n_rows)
        if config.shape == (N, 2):
            initial_configs.append(config)
    
    # Square grid
    initial_configs.append(_square_grid())
    
    # Perturbed hex grids
    for n_rows in [14, 15, 16]:
        config = _hex_grid(n_rows)
        if config.shape == (N, 2):
            perturbed = config + np.random.randn(N, 2) * 0.01
            perturbed = np.clip(perturbed, -0.5, 0.5)
            initial_configs.append(perturbed)
    
    # Time budget allocation
    n_configs = len(initial_configs)
    time_per_config = (DEADLINE - start_time - 30.0) / n_configs  # Reserve 30s for final polishing
    
    for config in initial_configs:
        if time.time() > DEADLINE - 35.0:
            break
        
        # Force-directed relaxation
        fd_deadline = time.time() + max(5.0, time_per_config * 0.4)
        fd_centers, fd_r = _force_directed(config, steps=150, deadline=fd_deadline)
        
        # Simulated annealing
        sa_deadline = time.time() + max(5.0, time_per_config * 0.35)
        sa_centers, sa_r = _sim_annealing(fd_centers, n_iter=3000, t_start=0.004, t_end=0.0001, deadline=sa_deadline)
        
        # Coordinate descent
        cd_deadline = time.time() + max(3.0, time_per_config * 0.25)
        cd_centers, cd_r = _coordinate_descent(sa_centers, deadline=cd_deadline)
        
        if cd_r > best_radius:
            best_radius = cd_r
            best_centers = cd_centers.copy()
    
    # Final polishing with more aggressive coordinate descent
    if best_centers is not None and time.time() < DEADLINE - 5.0:
        final_deadline = DEADLINE - 2.0
        final_centers, final_r = _coordinate_descent(best_centers, deadline=final_deadline)
        if final_r > best_radius:
            best_radius = final_r
            best_centers = final_centers
    
    # Ensure we always return a valid configuration
    if best_centers is None or best_centers.shape != (N, 2):
        best_centers = _square_grid()
    
    return best_centers
