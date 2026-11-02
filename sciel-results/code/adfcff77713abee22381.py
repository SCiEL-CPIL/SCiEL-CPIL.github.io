"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _hex_init(n):
    """Generate a hexagonal lattice of n points."""
    # Approximate number of points in a hexagonal cluster of radius R
    # n ~ (3/2) R^2 + ...
    # For n=200, R ~ sqrt(400/3) ~ 11.5
    # Let's generate a grid and crop.
    
    # Hex lattice vectors
    a = 1.0 # lattice spacing
    b = a * np.sqrt(3) / 2
    
    # Generate points
    pts = []
    # Range needs to be large enough to get n points
    # Area per point ~ a^2 * sqrt(3)/2
    # Side length of square ~ sqrt(n)
    # Let's just generate a square grid of hex points
    size = int(np.ceil(np.sqrt(n / 0.8))) # 0.8 is packing density factor approx
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            x = i * a + (j % 2) * (a / 2)
            y = j * b
            pts.append([x, y])
            
    pts = np.array(pts)
    
    # Select n points closest to the center (0,0)
    dists = np.sum(pts**2, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx[:n]]
    
    # Center the points
    center = np.mean(pts, axis=0)
    pts -= center
    
    # Scale to fit in [-0.5, 0.5] with some margin
    # The max coordinate is roughly size * a
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        # We want the points to be well inside initially
        # Target max coord ~ 0.4
        scale = 0.4 / max_coord
        pts *= scale
        
    return pts

def _score(pts):
    """Calculate the maximum radius."""
    if len(pts) == 0:
        return 0.0
    
    # Pairwise distances
    # To save memory/time for N=200, we can do it in chunks or just use broadcasting if memory allows
    # 200x200 matrix is small (40k entries)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Boundary distance
    # Distance to boundary for each point
    # Square is [-0.5, 0.5]
    # dist to boundary = 0.5 - max(|x|, |y|)
    max_abs_coord = np.max(np.abs(pts), axis=1)
    min_bound_dist = np.min(0.5 - max_abs_coord)
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    # 1. Initialization
    pts = _hex_init(N)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # 2. Simulated Annealing
    # Parameters
    T = 0.01 # Initial temperature
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    # Number of steps depends on time
    
    # Pre-calculate some things? No, dynamic.
    
    # To speed up distance calculation, we can use a neighbor list or just rely on NumPy speed.
    # For N=200, full O(N^2) is ~1ms. 
    # If we do 1 step per 1ms, we can do ~100,000 steps in 100s.
    
    # Let's try to be smarter.
    # We will move one point at a time.
    
    current_pts = best_pts.copy()
    current_score = best_score
    
    step = 0
    while time.time() < deadline:
        # Move a random point
        i = np.random.randint(N)
        old_pt = current_pts[i].copy()
        
        # Random move
        # Step size should be related to T
        move_scale = T * 2.0
        move = np.random.normal(0, move_scale, 2)
        new_pt = old_pt + move
        
        # Ensure new point is within bounds? 
        # If we push it out, the score drops significantly.
        # It's better to let the score function handle it, but maybe clip to [-0.5, 0.5] to avoid huge penalties?
        # No, if we clip, we bias the search towards the boundary.
        # The score function will penalize being outside by reducing the boundary distance term to negative?
        # My _score function returns min(pair/2, bound). If bound is negative, score is negative.
        # This is fine.
        
        current_pts[i] = new_pt
        
        new_score = _score(current_pts)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta > 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(delta / T)
            if np.random.random() < np.exp(delta / T):
                current_score = new_score
            else:
                # Reject
                current_pts[i] = old_pt
        
        # Cool down
        T *= alpha
        
        step += 1
        # Throttle if too fast? No, we want max steps.
        
    # 3. Local Polish (Optional)
    # If we have time, we can try a simple local search
    # But SA usually does a decent job.
    
    return best_pts
# EVOLVE-BLOCK-END
