import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the actual score: (max_dist / min_dist)^2 = max_sq / min_sq."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def fibonacci_sphere(n_points):
    """Generate n_points on a sphere using the Fibonacci spiral method."""
    if n_points == 0:
        return np.array([]).reshape(0, 3)
    if n_points == 1:
        return np.array([[0.0, 0.0, 1.0]])
    
    i = np.arange(n_points)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    cos_theta = 1.0 - (2.0 * i + 1.0) / n_points
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    theta = np.mod(phi * i, 2.0 * np.pi)
    
    x = np.cos(theta) * sin_theta
    y = np.sin(theta) * sin_theta
    z = cos_theta
    return np.column_stack((x, y, z))

def objective_smooth(x, tau=20.0):
    """
    Smooth approximation of log(max_sq / min_sq) for optimization.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft max
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    # Generate initial candidates
    candidates = []
    
    # 1. Grid-based initialization
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    candidates.append(init_grid.flatten())
    
    # 2. Fibonacci sphere initialization
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 3. Random starts at various scales
    rng = np.random.default_rng(42)
    for scale in [1.0, 2.0, 0.5]:
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * scale
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout:
            break

    # Optimization loop with multiple methods
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout:
            break
            
        try:
            # Try L-BFGS-B
            res = minimize(
                objective_smooth, 
                init_x, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.fun < best_score:
                pts = res.x.reshape(N, D)
                true_score = raw_score(pts)
                if true_score < best_score:
                    best_score = true_score
                    best_points = pts
                    
        except Exception:
            pass
            
        if time.time() - start_time > timeout:
            break

    # If optimization didn't find anything better, fall back to the grid
    if best_points is None:
        best_points = init_grid
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
