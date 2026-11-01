import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50

def _compute_radius(points):
    """
    Computes the radius r for a given set of points.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    Returns 0.0 if invalid.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0
    
    # Compute pairwise distances using KDTree for efficiency
    tree = cKDTree(points)
    # Query nearest neighbors (2nd nearest, since 1st is self)
    dists, _ = tree.query(points, k=2)
    min_pair_dist = np.min(dists[:, 1])
    
    # Compute distance to walls
    wall_dists = np.min(np.stack([0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1])], axis=1), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r if r > 0 else 0.0

def _sa_search(points, start_time, deadline, T_init=0.02, T_min=1e-7, alpha=0.9995, step_init=0.03):
    """
    Simulated Annealing search starting from initial points.
    """
    current_points = points.copy()
    current_score = _compute_radius(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = T_init
    step_size = step_init
    iters = 0
    
    while time.time() - start_time < deadline:
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Generate a Gaussian move
        move = np.random.normal(0, step_size, 2)
        new_points = current_points.copy()
        new_points[idx] += move
        
        # Keep in bounds
        new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
        
        new_score = _compute_radius(new_points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        # Cooling
        if T > T_min:
            T *= alpha
        
        iters += 1
        
        # Adaptive step size: reduce if stuck, increase if improving
        if iters % 500 == 0:
            if current_score > best_score:
                step_size = max(0.005, step_size * 0.95)
            else:
                step_size = min(0.1, step_size * 1.05)
    
    return best_points, best_score

def _local_refine(points, max_iter=200):
    """
    Local refinement using scipy.optimize.minimize with SLSQP.
    Maximizes radius by minimizing -radius.
    """
    def objective(x):
        pts = x.reshape(N, 2)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 1.0
        r = _compute_radius(pts)
        return -r
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(objective, points.flatten(), method='SLSQP', 
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10})
        refined_points = result.x.reshape(N, 2)
        refined_score = _compute_radius(refined_points)
        
        if refined_score > 0:
            return refined_points, refined_score
    except Exception:
        pass
    
    return points, _compute_radius(points)

def _make_initial_grid(seed):
    """
    Create a jittered 7x7 grid + 1 extra point as initial configuration.
    """
    rng = np.random.RandomState(seed)
    
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Add 50th point at center with some randomness
    p50 = np.array([0.0, 0.0]) + rng.normal(0, 0.02, 2)
    
    points = np.vstack([grid, p50])
    
    # Jitter to break symmetry
    points += rng.normal(0, 0.005, points.shape)
    
    # Clip to valid range
    points = np.clip(points, -0.49, 0.49)
    
    # Shuffle order
    idx = rng.permutation(N)
    points = points[idx]
    
    return points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    # Multi-start SA with different seeds
    n_starts = 8
    time_per_start = (deadline - start_time) / (n_starts + 1)
    
    for s in range(n_starts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 10:
            break
        
        seed = 42 + s * 1000
        initial_points = _make_initial_grid(seed)
        
        # Run SA for a portion of the time
        sa_deadline = min(deadline - 20, start_time + elapsed + time_per_start * 0.7)
        sa_points, sa_score = _sa_search(initial_points, start_time, sa_deadline)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # Final local refinement on the best result
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            refined_points, refined_score = _local_refine(best_points, max_iter=100)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points
    
    if best_points is None:
        # Fallback: return a basic grid
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        p50 = np.array([0.0, 0.0])
        best_points = np.vstack([grid, p50])
    
    return best_points
