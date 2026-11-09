"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _min_distance(pts):
    """Compute the minimum distance between any two circles and to the walls."""
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(pts), 0.5 + np.abs(pts))  # not quite right
    # Actually: distance from point to boundary of [-0.5, 0.5]^2
    # For a point (x,y), distance to nearest wall is min(0.5-x, x+0.5, 0.5-y, y+0.5)
    # = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y) but since |x|<=0.5 and |y|<=0.5:
    # distance to left wall = x + 0.5, right = 0.5 - x, bottom = y + 0.5, top = 0.5 - y
    # min of these = min(0.5 - |x|, 0.5 - |y|)
    wall_min = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    
    # Pairwise distances
    n = len(pts)
    # Use scipy cdist for efficiency
    from scipy.spatial.distance import cdist
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    pair_min = dists.min()
    
    return min(wall_min.min(), pair_min)


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    np.random.seed(42)
    
    # Initial configuration: slightly perturbed grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])[:N].astype(float)
    
    # Add small random perturbation to break symmetry
    pts += np.random.randn(N, 2) * 0.001
    
    # Ensure points stay within bounds
    pts = np.clip(pts, -0.499, 0.499)
    
    best_pts = pts.copy()
    best_r = _min_distance(pts) / 2.0
    
    # Simulated annealing parameters
    T = 0.005  # initial temperature
    T_min = 1e-6
    cooling = 0.998
    max_iter = 500000
    
    current_r = best_r
    current_pts = pts.copy()
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose a random point to move
        idx = np.random.randint(N)
        
        # Propose a move: push away from nearest neighbor or toward center
        # Find nearest neighbor
        dists = np.linalg.norm(current_pts - current_pts[idx], axis=1)
        dists[idx] = np.inf
        nn_idx = np.argmin(dists)
        
        # Move direction: away from nearest neighbor
        if nn_idx != idx:
            direction = current_pts[idx] - current_pts[nn_idx]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
        else:
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
        
        # Step size proportional to temperature
        step_size = T * 2.0
        move = direction * step_size
        
        # Also add small random component
        move += np.random.randn(2) * T * 0.5
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + move
        
        # Keep within bounds
        new_pts[idx] = np.clip(new_pts[idx], -0.499, 0.499)
        
        # Compute new radius
        new_r = _min_distance(new_pts) / 2.0
        
        # Accept or reject
        delta = new_r - current_r
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_pts = new_pts
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final: ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
