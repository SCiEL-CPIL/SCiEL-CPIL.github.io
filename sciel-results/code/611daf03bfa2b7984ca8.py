# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _neg_log_score(flat_pts, n=16):
    """Negative log of score for minimization."""
    pts = flat_pts.reshape(n, 2)
    s = _compute_score(pts)
    if s <= 0:
        return 1e10
    return -np.log(s)


def _local_refine(points, deadline, max_powell_iters=200):
    """Local refinement using Powell method."""
    n = 16
    current_score = _compute_score(points)
    flat = points.flatten()
    
    try:
        result = minimize(_neg_log_score, flat, method='Powell', 
                         args=(n,),
                         options={'maxiter': max_powell_iters, 'xtol': 1e-12, 'ftol': 1e-14})
        pts = result.x.reshape(n, 2)
        score = _compute_score(pts)
        if score > current_score:
            return pts, score
    except Exception:
        pass
    
    return points, current_score


def _simulated_annealing(points, deadline, seed=42):
    """Simulated annealing to escape local minima."""
    n = 16
    rng = np.random.RandomState(seed)
    
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    # Temperature schedule
    T_start = 0.5
    T_end = 1e-6
    n_iter = 5000
    cooling_rate = (T_end / T_start) ** (1.0 / n_iter)
    
    T = T_start
    step_size = 0.3
    
    for i in range(n_iter):
        if time.time() > deadline - 30:
            break
        
        # Propose a move: perturb one or two points
        n_perturb = rng.randint(1, 4)  # 1-3 points
        idx = rng.choice(n, size=n_perturb, replace=False)
        
        new_points = current.copy()
        for idx_i in idx:
            new_points[idx_i] += rng.normal(0, step_size, size=2)
        
        new_score = _compute_score(new_points)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_score > current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta = -np.log(max(new_score, 1e-15)) + np.log(max(current_score, 1e-15))
            if delta < 0:
                delta = 0
            prob = np.exp(-delta / T) if T > 1e-10 else 0
            if rng.random() < prob:
                current = new_points
                current_score = new_score
        
        T *= cooling_rate
    
    return best, best_score


def _generate_initial_configs(n=16, n_configs=8, seed=0):
    """Generate diverse initial configurations."""
    rng = np.random.RandomState(seed)
    configs = []
    
    # 1. 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    grid = (grid - grid.mean(axis=0)) / (grid.std() + 1e-10)
    configs.append(grid)
    
    # 2. Hexagonal lattice
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col + (0.5 if row % 2 else 0)
            y = row * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / (hex_pts.std() + 1e-10)
    configs.append(hex_pts)
    
    # 3-8. Random with varying densities
    for k in range(6):
        pts = rng.uniform(-1, 1, size=(n, 2))
        pts = (pts - pts.mean(axis=0)) / (pts.std() + 1e-10)
        configs.append(pts)
    
    return configs


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit
    
    best_points = None
    best_score = -1.0
    
    # Generate initial configurations
    configs = _generate_initial_configs(n=n, n_configs=8, seed=42)
    
    # For each config: SA -> local refine -> track best
    for config_idx, config in enumerate(configs):
        if time.time() > deadline - 60:
            break
        
        # Simulated annealing
        sa_points, sa_score = _simulated_annealing(config, deadline, seed=42 + config_idx)
        
        # Local refinement
        refined_points, refined_score = _local_refine(sa_points, deadline, max_powell_iters=150)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
    
    # If time remains, do additional SA restarts from best point
    if best_points is not None and time.time() < deadline - 90:
        n_extra = 3
        for k in range(n_extra):
            if time.time() > deadline - 60:
                break
            
            # Perturb the best point
            rng = np.random.RandomState(1000 + k)
            perturbed = best_points + rng.normal(0, 0.2, size=(n, 2))
            
            sa_points, sa_score = _simulated_annealing(perturbed, deadline, seed=2000 + k)
            refined_points, refined_score = _local_refine(sa_points, deadline, max_powell_iters=100)
            
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points.copy()
    
    # Final polish on best
    if best_points is not None and time.time() < deadline - 30:
        final_points, final_score = _local_refine(best_points, deadline, max_powell_iters=300)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Ensure valid output
    if best_points is None:
        best_points = configs[0]
    
    assert best_points.shape == (16, 2), f"Invalid shape: {best_points.shape}"
    assert np.all(np.isfinite(best_points)), "Non-finite values in points"
    
    return best_points


# EVOLVE-BLOCK-END
