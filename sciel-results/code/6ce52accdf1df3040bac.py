# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
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


def _sqp_maximize_min_dist(points, max_dist_sq, deadline, maxiter=100):
    """
    SQP-style approach: maximize the minimum distance subject to all pairwise distances <= max_dist.
    We formulate this as minimizing -min_dist with constraint that max_dist <= max_dist_sq.
    Since scipy doesn't have built-in SQP, we use SLSQP with penalty for constraint violation.
    """
    n = 16
    
    def objective(flat_pts):
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        # We want to maximize d_min subject to d_max <= max_dist
        # Penalty if constraint violated
        if d_max > max_dist_sq:
            penalty = 1e6 * (d_max - max_dist_sq)
        else:
            penalty = 0
        return -d_min + penalty
    
    # Constraints: all pairwise distances squared <= max_dist_sq
    def constraints(flat_pts):
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, metric='sqeuclidean')
        # Return 0 for satisfied constraints, negative for violated
        return max_dist_sq - d
    
    flat = points.flatten()
    
    try:
        result = minimize(
            objective,
            flat,
            method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': lambda x: max_dist_sq - pdist(x.reshape(n,2), metric='sqeuclidean').max()}],
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        if result.success:
            pts = result.x.reshape(n, 2)
            return pts
    except Exception:
        pass
    
    return points


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
    
    # Phase 1: SA + local refine for each config
    for config_idx, config in enumerate(configs):
        if time.time() > deadline - 120:
            break
        
        # Simulated annealing
        sa_points, sa_score = _simulated_annealing(config, deadline, seed=42 + config_idx)
        
        # Local refinement
        refined_points, refined_score = _local_refine(sa_points, deadline, max_powell_iters=150)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
    
    # Phase 2: Bisection on max_dist with SQP inner loop
    if best_points is not None and time.time() < deadline - 90:
        # Current best has some d_max
        d = pdist(best_points)
        current_d_max = np.max(d)
        current_d_min = np.min(d)
        
        # Try to improve by constraining max_dist and maximizing min_dist
        # Bisection: try smaller max_dist values
        lo = current_d_max * 0.8
        hi = current_d_max * 1.2
        
        # First, check if we can achieve a better ratio with current d_max
        # Then try to reduce d_max while maintaining d_min
        
        for bisection_iter in range(8):
            if time.time() > deadline - 60:
                break
            
            mid = (lo + hi) / 2
            max_dist_sq = mid ** 2
            
            # Use SQP to maximize min_dist subject to max_dist <= mid
            sqp_points = _sqp_maximize_min_dist(best_points, max_dist_sq, deadline, maxiter=50)
            
            # Refine with Powell
            refined, score = _local_refine(sqp_points, deadline, max_powell_iters=100)
            
            if score > best_score:
                best_score = score
                best_points = refined.copy()
            
            # Check if the constraint was binding (i.e., max_dist was close to mid)
            d_check = pdist(refined)
            d_max_check = np.max(d_check)
            
            if d_max_check < mid * 0.95:
                # Constraint not binding, we can try smaller max_dist
                hi = mid
            else:
                # Constraint binding or violated, try larger max_dist
                lo = mid
    
    # Phase 3: If time remains, do additional SA restarts from best point
    if best_points is not None and time.time() < deadline - 60:
        n_extra = 3
        for k in range(n_extra):
            if time.time() > deadline - 30:
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
    if best_points is not None and time.time() < deadline - 15:
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
