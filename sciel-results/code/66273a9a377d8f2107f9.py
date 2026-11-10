import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """Compute the maximum common radius for given circle centers."""
    # Boundary constraint
    min_bound = 0.5 - np.max(np.abs(positions))
    
    # Pairwise distance constraint
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_bound, min_dist / 2.0)

def objective(x):
    """Objective function for Nelder-Mead minimization."""
    positions = x.reshape(N, 2)
    
    # Penalty for being outside [-0.5, 0.5]
    penalty = 0.0
    # Vectorized penalty calculation
    out_of_bounds = np.maximum(0, np.abs(positions) - 0.5)
    penalty = np.sum(out_of_bounds**2)
    
    r = compute_radius(positions)
    return -r + 100.0 * penalty

def generate_initial(N):
    """Generate a jittered hexagonal lattice initialization."""
    # Approximate spacing for 50 circles in unit square
    # Area per circle ~ 1/50 = 0.02. Hex area ~ s^2 * sqrt(3)/2.
    # s ~ sqrt(0.02 * 2 / 1.732) ~ sqrt(0.023) ~ 0.15
    spacing = 0.145
    
    points = []
    row = 0
    y = -0.5
    while y < 0.5:
        x_offset = (row % 2) * spacing / 2
        x = -0.5 + x_offset
        while x < 0.5:
            points.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
        
    points = np.array(points)
    
    # Add jitter to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    
    # Select N points
    if len(points) >= N:
        # Pick points closest to the center to ensure they are well-distributed
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        points = points[indices]
    else:
        # If not enough, pad with random points (shouldn't happen with spacing 0.145)
        while len(points) < N:
            points = np.vstack([points, np.random.uniform(-0.4, 0.4, 2)])
            
    # Center the points
    points -= np.mean(points, axis=0)
    
    # Scale to fit inside with margin
    max_abs = np.max(np.abs(points))
    if max_abs > 0.49:
        points *= 0.49 / max_abs
        
    return points

def polish(positions, time_limit=5.0):
    """
    Use Nelder-Mead to polish the positions.
    """
    start_time = time.time()
    x0 = positions.flatten()
    
    # Limit iterations based on time
    # Nelder-Mead iterations are expensive.
    # Estimate: 1000 iterations might take ~1-2s?
    # Let's allow up to 2000 iterations or until time runs out.
    max_iter = 2000
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.x is not None:
            pos = res.x.reshape(N, 2)
            # Ensure validity
            if np.all(np.abs(pos) <= 0.5) and np.all(np.isfinite(pos)):
                return pos
    except Exception:
        pass
        
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    np.random.seed(42)
    
    best_positions = None
    best_radius = -1
    
    # Try multiple initializations to escape local minima
    # Nelder-Mead is local, so a good start is crucial.
    
    num_attempts = 3
    
    for i in range(num_attempts):
        if time.time() > deadline - 30:
            break
            
        # Generate initial positions
        # Add more jitter for subsequent attempts
        jitter_scale = 0.01 + i * 0.01
        old_rand_state = np.random.get_state()
        np.random.seed(42 + i)
        
        # Temporarily modify generate_initial to accept jitter?
        # Or just call it and rely on the seed change?
        # generate_initial uses np.random.normal.
        
        positions = generate_initial(N)
        
        # Restore state
        np.random.set_state(old_rand_state)
        
        current_radius = compute_radius(positions)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_positions = positions.copy()
            
        # Run Nelder-Mead polish
        time_left = deadline - time.time()
        if time_left > 10:
            polished = polish(positions, time_limit=time_left - 5)
            polished_radius = compute_radius(polished)
            
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_positions = polished.copy()
                
    if best_positions is None:
        best_positions = generate_initial(N)
        
    return best_positions
