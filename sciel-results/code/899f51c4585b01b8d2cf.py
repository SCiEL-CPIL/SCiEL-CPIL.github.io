# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_ratio(points: np.ndarray) -> float:
    """Compute R = (d_min / d_max)^2 for the given points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _soft_max(d, k):
    """Smooth approximation of max(d) using soft-max."""
    return np.log(np.sum(np.exp(k * d))) / k


def _soft_min(d, k):
    """Smooth approximation of min(d) using soft-min."""
    return -_soft_max(-d, k)


def _objective(points, k):
    """Objective: minimize soft_max(d) - lambda * soft_min(d) to maximize ratio."""
    d = pdist(points)
    # We want to maximize d_min / d_max, equivalent to minimizing d_max / d_min
    # Or maximizing log(d_min) - log(d_max)
    # Using soft approximations:
    # max_d approx = soft_max(d, k)
    # min_d approx = soft_min(d, k)
    # We want to maximize min_d / max_d
    # Let's try minimizing: max_d - alpha * min_d
    # Or better: minimize (max_d / min_d) but that's non-smooth and can blow up.
    # Let's stick to minimizing max_d subject to min_d >= constraint, 
    # but since we are doing SCP, let's use a penalty method.
    
    # Actually, the prompt suggests: minimize max distance subject to min distance >= lb.
    # Here we just want a good objective for local refinement.
    # Let's use: minimize max_d - min_d (spread) is not right.
    # We want max ratio. 
    # Let's minimize: max_d / min_d.
    # To make it smooth: minimize soft_max(d, k) / soft_min(d, k).
    # This is positive and convex-ish? No, ratio is not convex.
    
    # Alternative: maximize log(min_d) - log(max_d).
    # Approx: log(soft_min(d, k)) - log(soft_max(d, k)).
    # This is smooth.
    
    smax = _soft_max(d, k)
    smin = _soft_min(d, k)
    
    if smin <= 0:
        return 1e10
    if smax <= 0:
        return 1e10
        
    # Minimize negative of log ratio
    return -(np.log(smin) - np.log(smax))


def _refine(points, time_left, max_iter=50):
    """Local refinement using gradient-based optimization."""
    if time_left <= 0:
        return points
        
    n = points.shape[0]
    best_points = points.copy()
    best_score = _compute_ratio(points)
    
    # Try increasing k (sharpness)
    k_values = [50, 100, 200, 400]
    
    for k in k_values:
        if time_left <= 10:
            break
            
        def obj(x):
            p = x.reshape(-1, 2)
            return _objective(p, k)
        
        try:
            res = minimize(obj, points.flatten(), method='BFGS', 
                           options={'maxiter': max_iter, 'gtol': 1e-8})
            
            if res.success:
                p_new = res.x.reshape(-1, 2)
                score_new = _compute_ratio(p_new)
                if score_new > best_score:
                    best_score = score_new
                    best_points = p_new
        except Exception:
            pass
            
    return best_points


def _random_start(n, d, seed):
    np.random.seed(seed)
    return np.random.randn(n, d)


def _circle_start(n, d, seed):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.sqrt(np.random.uniform(0.1, 1.0, n))
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack((x, y))


def _grid_start(n, d, seed):
    # Arrange 16 points in a 4x4 grid with some jitter
    np.random.seed(seed)
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    points += 0.1 * np.random.randn(n, d)
    return points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    n = 16
    d = 2
    
    best_points = None
    best_score = -1.0
    
    # Strategy: Multiple restarts with different initializations, followed by local refinement.
    # We will use a combination of random, circle, and grid starts.
    
    num_restarts = 50  # We will adjust based on time
    
    # Pre-compute a few good initializations
    initializations = []
    
    # 1. Circle starts with different radii and rotations
    for r in [0.5, 0.8, 1.0, 1.5]:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        initializations.append(np.column_stack((x, y)))
        
    # 2. Grid starts with jitter
    for seed in [0, 1, 2]:
        initializations.append(_grid_start(n, d, seed))
        
    # 3. Random starts
    for seed in [0, 1, 2, 3, 4]:
        initializations.append(_random_start(n, d, seed))
        
    # Shuffle initializations
    np.random.seed(42)
    np.random.shuffle(initializations)
    
    for i, init_points in enumerate(initializations):
        if time.time() >= deadline:
            break
            
        # Refine this initialization
        refined = _refine(init_points, deadline - time.time())
        
        score = _compute_ratio(refined)
        
        if score > best_score:
            best_score = score
            best_points = refined
            
        # If we have found a very good solution, we might want to spend more time polishing it
        # But for now, let's just keep going to see if we find something better.
        
        # Check time
        if time.time() >= deadline:
            break
        
        # If we have used up a significant portion of time, reduce the number of remaining restarts
        elapsed = time.time() - start_time
        if elapsed > 300:
            # If we've spent 5 minutes, we should be wrapping up
            if i > 20:
                break

    if best_points is None:
        # Fallback: return a random set
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        
    return best_points


# EVOLVE-BLOCK-END
