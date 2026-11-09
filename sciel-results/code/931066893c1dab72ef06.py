"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_score(pts):
    """Compute the largest common radius for given centers."""
    # pts shape (N, 2)
    # distance between pairs
    # Use scipy for cdist if available, else manual
    try:
        from scipy.spatial.distance import cdist
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min()
    except ImportError:
        # Fallback: manual computation (slower but works)
        min_pair = np.inf
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_pair:
                    min_pair = d
    
    # Distance to boundary: min_i min_k (1/2 - |c_ik|)
    # For each point, distance to nearest wall is min(0.5 - |x|, 0.5 - |y|)
    # Actually: min_k (1/2 - |c_ik|) means min over k=0,1 of (0.5 - |c_ik|)
    # So for point i, it's min(0.5 - abs(pts[i,0]), 0.5 - abs(pts[i,1]))
    wall_dists = np.min(np.abs(pts), axis=1)
    min_wall = 0.5 - wall_dists.max()
    
    return min(min_pair / 2.0, min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin for 150s limit
    
    # Start with a good initial packing
    # Use a hexagonal lattice pattern for better density
    m = int(np.ceil(np.sqrt(N)))
    # Generate hexagonal packing
    pts = []
    row = 0
    while len(pts) < N:
        x_offset = (row % 2) * 0.5 / m
        for col in range(m):
            if len(pts) >= N:
                break
            x = (col + 0.5 + x_offset) / m - 0.5
            y = (row + 0.5) / m - 0.5
            # Adjust for hexagonal spacing
            y = y * (m / (m * math.sqrt(3)/2))  # Scale y to make hexagonal
            pts.append([x, y])
        row += 1
    
    pts = np.array(pts[:N])
    
    # Better initial: use a grid with slight perturbation
    # Actually, let's use a proper hexagonal lattice
    # For 200 circles, try different configurations
    
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    # Try to improve using gradient-free optimization
    # Use a simple simulated annealing / hill climbing approach
    
    # First, let's try a few different initial configurations
    configs = []
    
    # Config 1: Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pts = np.array([(x, y) for y in ticks for x in ticks])[:N]
    configs.append(grid_pts.copy())
    
    # Config 2: Hexagonal lattice
    # Hexagonal packing: rows offset
    pts_hex = []
    # Calculate how many rows we need
    # In hexagonal packing, row spacing is s*sqrt(3)/2 where s is side
    # Let's try to fit 200 points
    for attempt in range(20):
        # Try different grid sizes
        cols = attempt + 13
        rows = int(np.ceil(N / cols))
        # Hexagonal spacing
        s = 1.0 / (cols + 0.5)  # Approximate
        pts_hex = []
        for r in range(rows):
            x_offset = (r % 2) * 0.5 * s
            for c in range(cols):
                if len(pts_hex) >= N:
                    break
                x = (c + 0.5) * s - 0.5 + x_offset
                y = (r + 0.5) * s * math.sqrt(3) / 2 - 0.5
                if abs(x) <= 0.5 and abs(y) <= 0.5:
                    pts_hex.append([x, y])
        if len(pts_hex) >= N:
            configs.append(np.array(pts_hex[:N]).copy())
    
    # Evaluate all configs
    for cfg in configs:
        if len(cfg) != N:
            continue
        score = _compute_score(cfg)
        if score > best_score:
            best_score = score
            best_pts = cfg.copy()
    
    # Now do local optimization on the best config
    # Use a simple coordinate descent / Nelder-Mead style approach
    
    # For time efficiency, use a simple iterative improvement
    # Move each point slightly to increase the minimum distance
    
    n_iter = 0
    max_iter = 50000
    step_size = 0.01
    
    current_pts = best_pts.copy()
    current_score = best_score
    
    # Use a simple hill climbing with random perturbations
    rng = np.random.default_rng(42)
    
    while time.time() < deadline and n_iter < max_iter:
        # Try to improve by moving a random point
        i = rng.integers(0, N)
        
        # Try small perturbations in random directions
        improved = False
        for _ in range(10):
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            new_pts = current_pts.copy()
            new_pts[i, 0] += dx
            new_pts[i, 1] += dy
            
            # Check bounds
            if np.any(np.abs(new_pts) > 0.5):
                continue
            
            score = _compute_score(new_pts)
            if score > current_score + 1e-12:
                current_pts = new_pts
                current_score = score
                improved = True
                break
        
        n_iter += 1
        
        # Gradually decrease step size
        if n_iter % 1000 == 0:
            step_size *= 0.9
    
    # Final polish: try to fix any boundary issues
    # Clamp points to be inside the square
    current_pts = np.clip(current_pts, -0.5 + 1e-10, 0.5 - 1e-10)
    
    # Return the best result
    return current_pts
# EVOLVE-BLOCK-END
