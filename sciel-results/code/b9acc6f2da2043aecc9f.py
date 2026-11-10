# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    # Objective: maximize log(min_dist) - log(max_dist)
    # This is equivalent to maximizing log(min/max) = log(R^0.5) = 0.5*log(R)
    # Maximizing this is equivalent to maximizing R.
    # We minimize the negative of this: -log(min_dist) + log(max_dist)
    
    def objective(x):
        # Check deadline
        if time.time() > deadline - 5:
            return 1e10  # Large penalty if we're out of time
        
        pts = x.reshape(n, 2)
        
        # Compute pairwise distances
        # Use pdist for efficiency
        dists = pdist(pts, 'euclidean')
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        # Handle edge cases
        if d_min <= 0 or d_max <= 0:
            return 1e10
        
        # We want to maximize min/max, so minimize -log(min/max)
        # = -log(min) + log(max)
        # Add a small regularization to prevent points from going to infinity or coinciding
        # Also, we should normalize the scale. The ratio is scale-invariant.
        # Let's fix the scale by requiring sum of squares = n (or similar) to avoid scale drift.
        # Actually, L-BFGS-B might drift in scale. 
        # Better approach: 
        # 1. Normalize points to unit norm (sum of squares = 1) inside the objective? No, that changes the geometry.
        # 2. Just let the optimizer handle it, but the objective is scale invariant, so it's a flat direction.
        # This flat direction can cause issues. 
        # Let's add a penalty for large scale or small scale to keep points bounded.
        # Or, better: fix one point at origin and another at (1,0)? No, that breaks symmetry.
        # Let's add a soft penalty for the sum of squared distances from the centroid being too large or small.
        # Centroid subtraction:
        centroid = np.mean(pts, axis=0)
        pts_c = pts - centroid
        scale = np.sum(pts_c**2)
        
        # We want scale to be around 1. 
        # Penalty if scale deviates from 1.
        # This helps stabilize the optimization.
        scale_penalty = (scale - 1.0)**2
        
        return -np.log(d_min) + np.log(d_max) + 0.1 * scale_penalty

    def objective_no_penalty(x):
        pts = x.reshape(n, 2)
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max <= 0:
            return 1e10
        return -np.log(d_min) + np.log(d_max)

    def score(pts):
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max)**2

    best_points = None
    best_score = -1.0

    # Initial guesses
    # 1. Hexagonal lattice
    # 2. Random
    # 3. Grid
    # 4. Ring
    
    initial_guesses = []
    
    # Hexagonal
    hex_pts = []
    # Generate hex grid points
    # Radius in hex units
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(q) <= 2 and abs(r) <= 2 and abs(s) <= 2:
                x = 1.5 * q
                y = np.sqrt(3) * (r + q / 2)
                hex_pts.append([x, y])
    # Take 16 closest to origin? Or just first 16?
    # Let's sort by distance to origin and take 16
    hex_pts = np.array(hex_pts)
    hex_pts -= np.mean(hex_pts, axis=0)
    dists_to_origin = np.sum(hex_pts**2, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    hex_pts = hex_pts[idx]
    # Normalize scale
    hex_pts /= np.sqrt(np.sum(hex_pts**2) / n)
    initial_guesses.append(hex_pts.flatten())
    
    # Random
    np.random.seed(42)
    rand_pts = np.random.randn(n, 2)
    rand_pts -= np.mean(rand_pts, axis=0)
    rand_pts /= np.sqrt(np.sum(rand_pts**2) / n)
    initial_guesses.append(rand_pts.flatten())
    
    # Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    grid_pts -= np.mean(grid_pts, axis=0)
    grid_pts /= np.sqrt(np.sum(grid_pts**2) / n)
    initial_guesses.append(grid_pts.flatten())
    
    # Ring
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    ring_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    initial_guesses.append(ring_pts.flatten())
    
    # Optimize each initial guess
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
            if res.success or res.fun < 1e9:
                pts = res.x.reshape(n, 2)
                # Re-center and normalize?
                # The objective included a scale penalty, so it should be near scale 1.
                # But let's check the score.
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception as e:
            pass

    # If we have time, try more random restarts
    restarts = 0
    while time.time() < deadline - 10:
        if restarts % 10 == 0 and best_points is not None:
            # Perturb best point
            x0 = best_points.flatten() + 0.1 * np.random.randn(n*2)
        else:
            x0 = np.random.randn(n*2)
            x0 = x0.reshape(n, 2)
            x0 -= np.mean(x0, axis=0)
            x0 /= np.sqrt(np.sum(x0**2) / n)
            x0 = x0.flatten()
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-7})
            if res.success or res.fun < 1e9:
                pts = res.x.reshape(n, 2)
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception as e:
            pass
        restarts += 1

    if best_points is None:
        # Fallback
        best_points = initial_guesses[0].reshape(n, 2)

    # Final check
    return best_points

# EVOLVE-BLOCK-END
