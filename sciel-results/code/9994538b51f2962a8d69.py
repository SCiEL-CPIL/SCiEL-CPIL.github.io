"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _score(centers):
    """Compute the valid radius for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    min_dist_to_wall = np.min(0.5 - np.abs(centers))
    if min_dist_to_wall < 0:
        return 0.0
    n = len(centers)
    min_pair_dist = np.inf
    for i in range(n):
        diffs = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if len(dists) > 0:
            min_pair_dist = min(min_pair_dist, np.min(dists))
    if not np.isfinite(min_pair_dist):
        return 0.0
    r = min(min_dist_to_wall, min_pair_dist / 2.0)
    return max(0.0, r)

def _hex_grid(n_rows):
    """Generate a hexagonal lattice configuration for N points."""
    n_cols = math.ceil(N / n_rows)
    d = 1.0 / (n_cols - 0.5)
    row_spacing = d * math.sqrt(3) / 2.0
    
    pts = []
    for row in range(n_rows):
        y = -0.5 + d/2.0 + row * row_spacing
        if row % 2 == 0:
            n_in_row = n_cols
            x_start = -0.5 + d/2.0
        else:
            n_in_row = n_cols - 1
            x_start = -0.5 + d
        
        for col in range(n_in_row):
            x = x_start + col * d
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

def _simulated_annealing(centers, n_iter=5000, t_start=0.01, t_end=0.0001, deadline=None):
    """Simulated annealing to optimize the packing."""
    centers = centers.copy()
    current_score = _score(centers)
    best_score = current_score
    best_centers = centers.copy()
    
    n = len(centers)
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        i = rng.randint(n)
        orig = centers[i].copy()
        
        step = t * 0.5
        delta = rng.randn(2) * step
        new_pos = orig + delta
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        centers[i] = new_pos
        new_score = _score(centers)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            delta_score = new_score - current_score
            if delta_score < -100 * t:
                centers[i] = orig
            elif rng.rand() < math.exp(delta_score / t):
                current_score = new_score
            else:
                centers[i] = orig
    
    return best_centers, best_score

def _local_optimization(centers, deadline=None):
    """Coordinate descent with adaptive step sizes."""
    centers = centers.copy()
    current_score = _score(centers)
    
    n = len(centers)
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    
    for step in step_sizes:
        if deadline is not None and time.time() > deadline:
            break
        
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if deadline is not None and time.time() > deadline:
                    break
                
                orig = centers[i].copy()
                best_pos = orig
                best_r = current_score
                
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = orig + np.array([dx, dy])
                        if new_pos[0] < -0.5 or new_pos[0] > 0.5:
                            continue
                        if new_pos[1] < -0.5 or new_pos[1] > 0.5:
                            continue
                        
                        centers[i] = new_pos
                        r = _score(centers)
                        if r > best_r:
                            best_r = r
                            best_pos = new_pos.copy()
                
                centers[i] = best_pos
                if not np.allclose(best_pos, orig):
                    improved = True
                    current_score = best_r
    
    return centers, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_centers = None
    best_radius = 0.0
    
    initial_configs = []
    
    for n_rows in range(12, 20):
        if time.time() > deadline - 10:
            break
        config = _hex_grid(n_rows)
        if config.shape == (N, 2):
            initial_configs.append(config)
    
    initial_configs.append(_square_grid())
    
    for n_rows in [14, 15, 16]:
        config = _hex_grid(n_rows)
        if config.shape == (N, 2):
            perturbed = config + np.random.randn(N, 2) * 0.005
            perturbed = np.clip(perturbed, -0.5, 0.5)
            initial_configs.append(perturbed)
    
    for config in initial_configs:
        if time.time() > deadline - 30:
            break
        r = _score(config)
        if r > best_radius:
            best_radius = r
            best_centers = config.copy()
    
    if best_centers is None:
        best_centers = _square_grid()
        best_radius = _score(best_centers)
    
    if time.time() < deadline - 20:
        sa_deadline = deadline - 15
        sa_centers, sa_score = _simulated_annealing(
            best_centers, n_iter=8000, t_start=0.008, t_end=0.00005, deadline=sa_deadline
        )
        if sa_score > best_radius:
            best_radius = sa_score
            best_centers = sa_centers
    
    if time.time() < deadline - 5:
        lo_deadline = deadline - 2
        lo_centers, lo_score = _local_optimization(best_centers, deadline=lo_deadline)
        if lo_score > best_radius:
            best_radius = lo_score
            best_centers = lo_centers
    
    if time.time() < deadline - 1:
        final_deadline = deadline - 0.5
        fc_centers, fc_score = _local_optimization(best_centers, deadline=final_deadline)
        if fc_score > best_radius:
            best_radius = fc_score
            best_centers = fc_centers
    
    return best_centers
# EVOLVE-BLOCK-END
