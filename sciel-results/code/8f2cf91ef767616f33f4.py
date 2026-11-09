import numpy as np
import time
from scipy.optimize import minimize

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts, test_points):
    """
    Estimate the covering radius by checking distances from test_points to the nearest point in pts.
    Returns the maximum of these minimum distances.
    """
    # pts: (N, 3), test_points: (M, 3)
    # Calculate squared distances
    # dist^2 = |y|^2 + |x|^2 - 2*y.x = 1 + 1 - 2*y.x = 2 - 2*y.x
    # We want max over y of min over x of dist(y, x)
    # This is equivalent to max over y of sqrt(2 - 2 * max_i (y . x_i))
    # Let v_y = max_i (y . x_i). Then dist = sqrt(2 - 2*v_y).
    # We want to maximize dist, which means minimizing v_y.
    # So covering radius is determined by the y that minimizes max_i (y . x_i).
    
    # Compute dot products: (M, N)
    dots = test_points @ pts.T
    # For each test point y, find the max dot product with any x_i
    max_dots = np.max(dots, axis=1)
    # The angular distance is arccos(max_dots)
    # To avoid numerical issues with arccos, we can just return the min max_dot (which corresponds to max distance)
    # Or return the max arccos.
    # Let's return the max angle in radians.
    angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
    return np.max(angles)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Test points for estimating covering radius
    # Use a dense set of points on the sphere
    M = 2000
    # Generate random points on sphere
    rng = np.random.default_rng(42)
    test_points = rng.normal(size=(M, 3))
    test_points = _normalize(test_points)
    
    # Also add some structured points? Random should be fine for a good estimate.
    
    best_pts = pts.copy()
    best_val = _covering_radius(best_pts, test_points)
    
    def objective(x_flat):
        # x_flat is 3N
        x = x_flat.reshape((N, 3))
        x = _normalize(x)
        return _covering_radius(x, test_points)
    
    # Use Powell's method which is derivative-free and robust for non-smooth functions
    # We will run it in a loop with multiple restarts if time permits
    # Or just one long run? Powell can be slow.
    # Let's try L-BFGS-B on a smoothed version?
    # Powell is safer.
    
    x0 = best_pts.flatten()
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 100, 'disp': False})
        if res.success or res.fun < best_val:
            best_pts = _normalize(res.x.reshape((N, 3)))
            best_val = res.fun
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < time_limit:
        # Try a few more local optimizations from slightly perturbed starts
        # Or refine the current best with a different method
        # Let's try a few random restarts with Nelder-Mead
        for _ in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            # Perturb best_pts
            noise = rng.normal(size=(N, 3), scale=0.1)
            x_start = _normalize(best_pts + noise).flatten()
            try:
                res = minimize(objective, x_start, method='Nelder-Mead', options={'maxiter': 200, 'disp': False})
                if res.fun < best_val:
                    best_val = res.fun
                    best_pts = _normalize(res.x.reshape((N, 3)))
            except Exception:
                pass
    
    # Final normalization
    return _normalize(best_pts)

# EVOLVE-BLOCK-START
# The code above is the complete program.
# We need to wrap it in the EVOLVE-BLOCK structure if required by the prompt format, 
# but the prompt says "Rewrite the program... Keep the program's entry-point function name(s)... Output the COMPLETE program".
# The current program has EVOLVE-BLOCK markers. I should probably keep the structure similar or just output the code.
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has:
# N = 130
# def construct(): ...
# I will output the complete file content.
# EVOLVE-BLOCK-END
