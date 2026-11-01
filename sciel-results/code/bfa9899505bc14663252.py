import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(points):
    """
    Calculate the maximum radius for the given points.
    points: (N, 2) array
    Returns: radius (float)
    """
    if not np.all(np.isfinite(points)):
        return -np.inf
        
    # Pairwise distances
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    x_dists = 0.5 - np.abs(points[:, 0])
    y_dists = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([x_dists, y_dists]))
    
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    return radius

def _soft_min_obj(points_flat, r_target, beta):
    """
    Smooth surrogate for maximizing min distance.
    We want to maximize r = min(d_ij/2, d_wall)
    This is equivalent to minimizing -r.
    We use a soft-min approximation: 
    -softmin(x) = (1/beta) * log(sum(exp(-beta * x_i)))
    where x_i are the distances (d_ij/2 and d_wall).
    Minimizing this pushes all x_i to be large and equal.
    """
    pts = points_flat.reshape(N, 2)
    
    if not np.all(np.isfinite(pts)):
        return 1e10
        
    # Pairwise distances (i < j)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Get upper triangle indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Wall distances
    wall_dists = np.concatenate([
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    ])
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min
    # To avoid overflow in exp, shift by min
    min_d = np.min(all_dists)
    if min_d < -100:
        return 1e10
        
    shifted = all_dists - min_d
    log_sum_exp = min_d + (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    
    # We want to maximize the min distance, so we minimize -log_sum_exp
    return -log_sum_exp

def _hex_seed():
    """
    Generate a hexagonal lattice seed.
    """
    a = 1.0
    v1 = np.array([a, 0])
    v2 = np.array([a/2, a*np.sqrt(3)/2])
    
    max_idx = 15
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            p = i * v1 + j * v2
            points.append(p)
            
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:N]]
    points -= np.mean(points, axis=0)
    
    max_extent = np.max(np.abs(points))
    if max_extent > 0:
        points *= (0.45 / max_extent)
        
    return points

def _grid_seed():
    """
    Generate a grid seed.
    """
    # 7x8 grid roughly
    rows = 7
    cols = 8
    x = np.linspace(-0.45, 0.45, cols)
    y = np.linspace(-0.45, 0.45, rows)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    points = points[:N]
    return points

def _random_seed():
    """
    Generate a random seed within the square.
    """
    return np.random.uniform(-0.45, 0.45, size=(N, 2))

def _polish_lbfgs(points, start_time, deadline):
    """
    Polish a configuration using L-BFGS-B on the soft-min objective.
    """
    if time.time() > deadline - 5:
        return points, _score(points)
        
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Adaptive beta: start low, increase
    betas = [10, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        def obj(x):
            return _soft_min_obj(x, 0, beta)
            
        try:
            res = minimize(obj, best_points.flatten(), method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            
            new_score = _score(res.x.reshape(N, 2))
            if new_score > best_score:
                best_score = new_score
                best_points = res.x.reshape(N, 2).copy()
                
        except Exception:
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Multi-start approach
    seeds = [_hex_seed(), _grid_seed(), _random_seed()]
    
    best_points = seeds[0].copy()
    best_score = _score(best_points)
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        current_score = _score(seed)
        if current_score > best_score:
            best_score = current_score
            best_points = seed.copy()
            
        # Polish each seed
        polished_pts, polished_score = _polish_lbfgs(seed, start_time, deadline)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
            
    # Final polish on the best found
    if time.time() < deadline - 5:
        final_pts, final_score = _polish_lbfgs(best_points, start_time, deadline)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
            
    # Ensure validity
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
