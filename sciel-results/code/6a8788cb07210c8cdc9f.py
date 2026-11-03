"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _vectorized_penalty(params):
    """
    Vectorized penalized objective.
    params: array of shape (3*N,) -> [x1, y1, r1, x2, y2, r2, ...]
    Returns: float objective (negative sum of radii + penalties)
    """
    n = len(params) // 3
    # Reshape to (n, 3) for easier handling
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Objective: minimize -sum(r)
    obj = -np.sum(r)
    
    # 1. Boundary Penalties
    # Distance to left: x - r, right: 1 - x - r, bottom: y - r, top: 1 - y - r
    # We want these to be >= 0.
    # Penalty is sum of (max(0, -d))^2 or similar. 
    # Using a smooth approximation or just squared violation.
    # Let's use: if d < 0, add penalty * d^2.
    
    d_left = x - r
    d_right = 1.0 - x - r
    d_bottom = y - r
    d_top = 1.0 - y - r
    
    # Only penalize violations
    # np.where(condition, x, y)
    # We want 0 if d >= 0, d^2 if d < 0
    # Actually, a smooth penalty like -log(d) is better for optimization but risky.
    # Quadratic penalty is standard.
    # Let's use a large weight.
    weight = 100.0
    
    viol_left = np.where(d_left < 0, d_left, 0)
    viol_right = np.where(d_right < 0, d_right, 0)
    viol_bottom = np.where(d_bottom < 0, d_bottom, 0)
    viol_top = np.where(d_top < 0, d_top, 0)
    
    obj += weight * (np.sum(viol_left**2) + np.sum(viol_right**2) + 
                     np.sum(viol_bottom**2) + np.sum(viol_top**2))
    
    # 2. Overlap Penalties
    # Calculate pairwise distances
    # Use broadcasting to get (n, n) distance matrix
    # dx = x[:, None] - x[None, :]
    # dy = y[:, None] - y[None, :]
    # dist = sqrt(dx^2 + dy^2)
    # min_dist = r[:, None] + r[None, :]
    # overlap = max(0, min_dist - dist)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    min_dist = r[:, None] + r[None, :]
    
    # We only care about i < j to avoid double counting and self-interaction
    # But vectorized sum over upper triangle is easier
    # Create a mask for upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Extract relevant parts
    dist_upper = dist[mask]
    min_dist_upper = min_dist[mask]
    
    viol_overlap = np.where(min_dist_upper > dist_upper, min_dist_upper - dist_upper, 0)
    
    obj += weight * np.sum(viol_overlap**2)
    
    return obj


def _hex_init():
    """
    Initialize circles in a hexagonal pattern.
    This is a denser packing than square grid.
    """
    # For N=26, a 5x5 grid is 25, so we need one extra.
    # Or 5 rows: 5, 5, 5, 5, 6?
    # Let's just generate a grid and perturb.
    # A better hex init:
    # Row 0: 5 circles
    # Row 1: 5 circles (offset)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 6 circles? No, that's too many.
    # 26 = 5*5 + 1.
    
    # Let's use a 5x5 grid and place the 26th in the center of a gap?
    # Or just use a 5x5 grid for the first 25 and place the 26th at (0.5, 0.5) with small radius?
    # No, that's bad.
    
    # Let's try a 6x5 grid? 30 circles. Remove 4.
    # Or just random perturbation of a square grid is safer and often works well with L-BFGS-B.
    
    # Let's stick to a structured grid but with hexagonal spacing where possible.
    # Approximate hex packing in square:
    # Vertical spacing: dy = 1.0 / (rows + 0.5) ?
    
    # Simple robust init: Random points, then let optimizer do the work.
    # But random init is slow to converge.
    
    # Let's use a 5x5 grid for 25 circles, and add the 26th at a random empty spot?
    # Better: 5x5 grid, all radii equal.
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            # Add small random jitter to break symmetry
            x = (i + 0.5) / side + np.random.uniform(-0.01, 0.01)
            y = (j + 0.5) / side + np.random.uniform(-0.01, 0.01)
            r = 0.5 / side * 0.9 # Slightly smaller to allow growth
            positions.append([x, y, r])
    
    # Add the 26th circle
    # Place it in the center of the square? (0.5, 0.5) is occupied by (2,2) in 0-indexed 5x5?
    # Indices 0..4. Center is (2,2).
    # Let's place the 26th at a random location and let it move.
    x26 = np.random.uniform(0.2, 0.8)
    y26 = np.random.uniform(0.2, 0.8)
    r26 = 0.1
    positions.append([x26, y26, r26])
    
    return np.array(positions)


def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    best_params = None
    best_score = np.inf # We are minimizing
    
    # We will try a few different initializations
    # 1. Grid Init
    # 2. Random Init
    
    np.random.seed(42)
    
    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    # But we need x >= r, x <= 1-r, etc.
    # L-BFGS-B handles bounds on variables, but not constraints like x >= r.
    # So we rely on penalties.
    # To help optimizer, we can set bounds on r to be reasonable.
    # Max possible radius is 0.5 (if only 1 circle). With 26, it's much smaller.
    # Approx area per circle: 1/26. Area = pi*r^2. r ~ sqrt(1/(26*pi)) ~ 0.11.
    # So r in [0.01, 0.2] is a safe bound.
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.01, 0.25)) # r
    
    trials = 10
    
    for trial in range(trials):
        if time.time() > deadline:
            break
            
        if trial == 0:
            init = _hex_init()
        else:
            # Random initialization
            # Avoid clustering by using random grid positions?
            # Just random uniform
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.random.uniform(0.05, 0.15, N)
            init = np.column_stack([x, y, r])
            
        params = init.flatten()
        
        try:
            result = minimize(
                _vectorized_penalty,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception as e:
            # print(f"Optimization failed: {e}")
            continue
            
    if best_params is None:
        # Fallback: Return a valid grid
        side = 5
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side * 0.9
                    positions.append([x, y, r])
        # Add 26th
        positions.append([0.5, 0.5, 0.01])
        return np.array(positions)
        
    # Extract and validate
    n = len(best_params) // 3
    xy = best_params[:2*n].reshape(n, 2)
    r = best_params[2*n:]
    
    result = np.column_stack([xy, r])
    
    # Final validation and clipping
    # Ensure circles are inside [0,1]^2
    for i in range(n):
        x, y, ri = result[i]
        # If ri is too large for position, shrink it
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if ri > max_r:
            ri = max_r * 0.99 # Slightly smaller to avoid boundary touch issues
            # Also need to check overlaps, but shrinking helps
        # Ensure positive radius
        ri = max(ri, 0.001)
        
        # Clip position to valid range given radius
        x = np.clip(x, ri, 1.0 - ri)
        y = np.clip(y, ri, 1.0 - ri)
        
        result[i] = [x, y, ri]
        
    return result
# EVOLVE-BLOCK-END
