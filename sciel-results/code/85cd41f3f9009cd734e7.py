# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    deadline = time.time() + 620  # safety margin before 630s kill

    # Helper: compute score R = (d_min / d_max)^2
    def score(points):
        if points.shape != (n, d):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Objective for minimizer: minimize -score
    def objective(flat):
        pts = flat.reshape(n, d)
        return -score(pts)

    # Initial guess: concentric ring configuration
    # 1 center + 5 inner ring + 10 outer ring? Or 4+4+4+4?
    # Let's try a structured initial guess based on known good configs.
    # A 4x4 grid is a decent start.
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    X, Y = np.meshgrid(x, y)
    points_init = np.column_stack((X.ravel(), Y.ravel()))
    points_init -= points_init.mean(axis=0)

    # Also try a circular arrangement
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    points_circle = np.column_stack((np.cos(angles), np.sin(angles)))

    # Select the better initial guess
    s1 = score(points_init)
    s2 = score(points_circle)
    if s1 > s2:
        best_points = points_init
    else:
        best_points = points_circle

    best_score_val = score(best_points)

    # Optimization loop
    # We will use L-BFGS-B which is good for large-scale bound-constrained problems.
    # We don't have explicit bounds, but we can normalize after each step or rely on the optimizer.
    # Since the problem is scale-invariant, we can fix the scale, e.g., mean distance = 1 or max distance = 1.
    # However, simple minimization of -score might drift scale. 
    # A better approach: optimize the coordinates directly.
    
    # Let's do a few restarts with different initial conditions if time permits.
    # Start with the best initial guess.
    
    start_time = time.time()
    
    # Run optimization
    try:
        res = minimize(
            objective,
            best_points.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < objective(best_points.ravel()):
            new_points = res.x.reshape(n, d)
            new_score = score(new_points)
            if new_score > best_score_val:
                best_points = new_points
                best_score_val = new_score
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 100:
        return best_points

    # Try more restarts with perturbed initial guesses
    np.random.seed(42)
    restarts = 10
    for i in range(restarts):
        if time.time() > deadline - 50:
            break
            
        # Perturb the current best or use a random start
        if i == 0:
            x0 = best_points + np.random.randn(n, d) * 0.1
        else:
            # Random start on unit circle or grid
            if i % 2 == 0:
                angles = np.linspace(0, 2 * np.pi, 16, endpoint=False) + np.random.randn(16) * 0.1
                x0 = np.column_stack((np.cos(angles), np.sin(angles)))
            else:
                x0 = np.random.randn(n, d)
        
        try:
            res = minimize(
                objective,
                x0.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < -best_score_val:
                new_points = res.x.reshape(n, d)
                new_score = score(new_points)
                if new_score > best_score_val:
                    best_points = new_points
                    best_score_val = new_score
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
