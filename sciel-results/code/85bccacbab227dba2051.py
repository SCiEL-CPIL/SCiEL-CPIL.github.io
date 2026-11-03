"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius(centers):
    """
    Compute the radius score: min of half min-distance and min boundary distance.
    Returns 0.0 if invalid.
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    # Distance to boundary for point (x,y) is min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
    # = 0.5 - max(|x|, |y|)
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    
    if min_boundary <= 0:
        return 0.0

    # Pairwise distances
    # Use broadcasting to compute distance matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r

def _initial_hex_grid(n):
    """
    Generate a hexagonal grid of n points inside the unit square.
    This provides a good starting point for local optimization.
    """
    # Estimate spacing
    # Area = 1.0. Density of hex packing ~ 0.9069.
    # Area per circle ~ 1/200 = 0.005.
    # Hex cell area A = 2*sqrt(3)*r^2.
    # 2*sqrt(3)*r^2 ~ 0.005 => r ~ 0.038.
    # Spacing dx = 2r ~ 0.076. dy = sqrt(3)*r ~ 0.066.
    
    # We want to generate roughly n points.
    # Let's aim for a grid slightly larger than needed and trim.
    # Or generate exactly n points by adjusting spacing.
    
    # Simple approach: generate a dense hex grid and take the first n?
    # Better: generate a hex grid that fits in [-0.5, 0.5]^2.
    
    # Let's try to fit 14x14 ~ 196 points.
    # Spacing x = 1.0 / 14 ~ 0.0714.
    # Spacing y = 0.0714 * sqrt(3)/2 ~ 0.0618.
    
    # Let's generate rows and cols.
    # Rows: y from -0.5 to 0.5.
    # Cols: x from -0.5 to 0.5.
    
    # We'll generate a grid of size 15x15 and then select 200 points 
    # that are best distributed or just take the first 200?
    # Taking the first 200 from a 15x15 grid (225 points) might bias towards one corner.
    # Better to generate a grid that covers the square and select 200 uniformly 
    # or just generate 200 points on a hex lattice centered in the square.
    
    # Let's calculate the required spacing to fit 200 points in a hex lattice 
    # that roughly fills the square.
    # Approximate number of points in a square of side L with spacing d:
    # N ~ (L/d)^2 * 2/sqrt(3) (since hex area is 2*sqrt(3)/3 * d^2? No, area per point is 2*sqrt(3)*r^2 = sqrt(3)/2 * d^2).
    # Area of square = 1.
    # N * sqrt(3)/2 * d^2 = 1 => d = sqrt(2 / (N * sqrt(3))).
    # For N=200: d = sqrt(2 / (200 * 1.732)) = sqrt(2 / 346.4) = sqrt(0.00577) ~ 0.076.
    
    d = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points
    points = []
    row = 0
    y = -0.5 + d * np.sqrt(3.0) / 2.0 # Start slightly inside to ensure boundary margin
    while y <= 0.5:
        x_offset = 0.0 if row % 2 == 0 else d / 2.0
        x = -0.5 + d / 2.0 + x_offset
        while x <= 0.5:
            points.append((x, y))
            x += d
        y += d * np.sqrt(3.0) / 2.0
        row += 1
        
    points = np.array(points)
    
    # If we have too many, we need to reduce to N.
    # If we have too few, we need to add more.
    # Given the estimation, we might have ~200-250 points.
    # Let's just take the first N. To be fair, we could randomize selection 
    # or use a subsampling that preserves distribution.
    # For now, taking the first N from a centered grid is a reasonable heuristic.
    # However, if the grid is offset, the first N might be biased.
    # Let's center the grid generation.
    
    # Alternative: Generate a dense grid and use a simple selection.
    # Let's stick to the generated list. If len(points) > N, we drop the last ones.
    # If len(points) < N, we need to handle it.
    
    if len(points) < N:
        # Add more points with smaller spacing? Or just duplicate?
        # Better to regenerate with smaller d.
        d *= 0.95
        points = []
        row = 0
        y = -0.5 + d * np.sqrt(3.0) / 2.0
        while y <= 0.5:
            x_offset = 0.0 if row % 2 == 0 else d / 2.0
            x = -0.5 + d / 2.0 + x_offset
            while x <= 0.5:
                points.append((x, y))
                x += d
            y += d * np.sqrt(3.0) / 2.0
            row += 1
        points = np.array(points)
        
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        # Pad with center points? This is bad.
        # Let's just pad with zeros and hope optimization fixes it? 
        # Or generate a square grid fallback.
        pass

    return points

def _objective_and_grad(x):
    """
    Objective function for Nelder-Mead.
    We want to MAXIMIZE the radius.
    Nelder-Mead MINIMIZES.
    So we minimize -radius.
    
    Note: The radius function is non-smooth (min of distances).
    Nelder-Mead is derivative-free, so it handles non-smooth functions better 
    than gradient-based methods, although it can still get stuck.
    """
    centers = x.reshape(N, 2)
    r = _radius(centers)
    # If invalid, return a large penalty
    if r <= 0:
        return 1000.0
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate initial hex grid
    centers = _initial_hex_grid(N)
    
    # Ensure shape is correct
    if centers.shape != (N, 2):
        # Fallback to random
        centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    # Flatten for optimizer
    x0 = centers.flatten()
    
    # 2. Run Nelder-Mead optimization
    # Maximize radius -> Minimize -radius
    
    # Set time limit for the optimizer
    # Nelder-Mead can be slow. We will run it in a loop with time checks 
    # or use the maxiter parameter.
    # Since we can't easily interrupt scipy.optimize.minimize mid-run,
    # we will use a conservative maxiter and check time.
    
    # Estimate iterations needed. 
    # N=200, 400 variables. NM is O(N_vars^2) per step? 
    # Actually, it's 401 simplex vertices. Each evaluation is O(N^2) for radius.
    # 401 * 200^2 = 16 million ops per iteration.
    # Python/NumPy overhead is high.
    # Let's say 1000 iterations max.
    
    # We can try to run it. If it takes too long, we return the best so far.
    # To be safe, we can run a few iterations manually or use a wrapper.
    # Scipy's minimize doesn't have a callback that can stop it easily 
    # without raising an exception.
    
    # Alternative: Implement a simple Nelder-Mead loop?
    # No, use scipy but with a tight maxiter.
    
    # Let's try 1000 iterations.
    # If it's too slow, we might timeout.
    # Let's monitor time.
    
    # A safer approach: Run minimize with a small maxiter, check time, 
    # and repeat if time permits.
    
    best_x = x0.copy()
    best_val = _objective_and_grad(best_x)
    
    # We'll run the optimizer in chunks or just once with a reasonable maxiter.
    # Given the 120s limit, and the cost of one evaluation (~1ms-5ms for N=200?),
    # 1000 iterations * 400 evals/iter = 400,000 evals.
    # 400,000 * 5ms = 2000s. This is WAY too slow.
    
    # Wait, how fast is _radius?
    # N=200. Distance matrix 200x200. 
    # np.sqrt(np.sum(diff**2, axis=2)).
    # This is very fast in NumPy. 
    # Let's profile mentally.
    # 200x200x2 array. ~ 80KB. 
    # Operations are vectorized. 
    # It should be < 1ms per call.
    # So 400,000 calls * 1ms = 400s. Still too slow for 1000 iterations.
    
    # We need to reduce the number of evaluations or make it faster.
    # Or use fewer iterations.
    
    # Let's try 100 iterations.
    # 100 * 400 = 40,000 evals.
    # 40,000 * 1ms = 40s. This is feasible.
    
    # Let's try maxiter=100.
    
    try:
        res = minimize(_objective_and_grad, x0, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return best_x.reshape(N, 2)
        
    # If time permits, run another round with tighter tolerance or more iterations
    # using the previous result as start.
    
    try:
        res2 = minimize(_objective_and_grad, best_x, method='Nelder-Mead', 
                        options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
        if res2.fun < best_val:
            best_val = res2.fun
            best_x = res2.x
    except Exception:
        pass
        
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
