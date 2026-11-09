# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize, differential_evolution

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def objective(x):
    """
    Objective function for minimization.
    We want to maximize (d_min / d_max)^2, so we minimize its negative.
    x is a flat array of shape (32,).
    """
    points = x.reshape(-1, 2)
    score = calculate_score(points)
    return -score

def hexagonal_init(n, scale=1.0, rotation=0.0):
    """Generate a hexagonal grid initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            points.append([x, y])
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    theta = rotation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * 0.9
    return pts

def random_init(n, seed=42):
    """Generate a random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    r = rng.rand(n) ** 0.5
    theta = rng.rand(n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def square_init(n, scale=1.0):
    """Generate a square grid initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            points.append([j * scale, i * scale])
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * 0.9
    return pts

def circle_init(n, scale=1.0):
    """Generate points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = scale * np.cos(angles)
    y = scale * np.sin(angles)
    return np.column_stack((x, y))

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 580
    
    best_result = None
    best_score = -1.0
    
    # 1. Try Differential Evolution for global search
    # This is more robust for finding the global optimum in non-convex landscapes
    # Bounds: [-2, 2] is sufficient since the metric is scale invariant
    # We can fix the scale to reduce dimensionality, but DE handles it.
    
    try:
        # Use a moderate population size to balance exploration and exploitation
        # maxiter is limited by time, but we set a reasonable number
        de_result = differential_evolution(
            objective, 
            bounds=[(-2, 2)] * (n * 2), 
            popsize=30, 
            maxiter=100, 
            tol=1e-4, 
            seed=42,
            init='sobol',
            polish=False # We will polish manually to save time
        )
        
        if de_result.success or de_result.fun < 0:
            points_de = de_result.x.reshape(-1, 2)
            score_de = -de_result.fun
            if score_de > best_score:
                best_score = score_de
                best_result = points_de
    except Exception as e:
        pass

    # 2. Local Polish with Powell
    # If we have a good candidate from DE or initial heuristics, polish it.
    # Also run Powell on some heuristic starts in case DE failed or got stuck.
    
    initializations = []
    initializations.append(square_init(n))
    initializations.append(hexagonal_init(n))
    initializations.append(circle_init(n))
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        initializations.append(random_init(n, seed=seed))
        
    # Add DE result if found
    if best_result is not None:
        initializations.insert(0, best_result)
        
    time_per_start = 5.0
    start_time = time.time()
    
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 30:
            break
            
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Powell', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            points_opt = res.x.reshape(-1, 2)
            score_opt = -res.fun
            
            if score_opt > best_score:
                best_score = score_opt
                best_result = points_opt
                
        except Exception as e:
            pass

    if best_result is None:
        best_result = square_init(n)
        best_score = calculate_score(best_result)
        
    return best_result

# EVOLVE-BLOCK-END
