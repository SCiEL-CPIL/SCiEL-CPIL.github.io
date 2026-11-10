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


def _soft_min_max_score(flat_pts, n=16, temp=0.05):
    """
    Smooth surrogate using soft-min and soft-max (log-sum-exp trick).
    Minimizes -log(soft_min/soft_max)^2 = 2*(log(soft_max) - log(soft_min))
    Uses normalized distances to make temp scale-invariant.
    """
    pts = flat_pts.reshape(n, 2)
    d = pdist(pts)
    if d.max() < 1e-15:
        return 1e10
    
    # Normalize distances to make temperature scale-invariant
    d_norm = d / d.max()
    
    # Soft max of normalized distances
    d_max_soft = temp * np.log(np.sum(np.exp(d_norm / temp)))
    # Soft min of normalized distances
    d_min_soft = -temp * np.log(np.sum(np.exp(-d_norm / temp)))
    
    ratio = d_min_soft / d_max_soft
    if ratio <= 0:
        return 1e10
    return -2.0 * np.log(ratio)


def _local_refine(points, deadline, max_powell_iters=200):
    """Local refinement using Powell method with multi-stage smooth surrogate, then exact polish."""
    n = 16
    current_score = _compute_score(points)
    flat = points.flatten()
    current_pts = points.copy()
    
    # Stage 1: Coarse smooth surrogate (larger temp)
    try:
        result1 = minimize(_soft_min_max_score, flat, method='Powell', 
                         args=(n, 0.1),
                         options={'maxiter': max_powell_iters // 2, 'xtol': 1e-8, 'ftol': 1e-10})
        pts1 = result1.x.reshape(n, 2)
        score1 = _compute_score(pts1)
        if score1 > current_score:
            current_score = score1
            current_pts = pts1.copy()
            flat = pts1.flatten()
    except Exception:
        pass
    
    # Stage 2: Fine smooth surrogate (smaller temp)
    if time.time() < deadline - 10:
        try:
            result2 = minimize(_soft_min_max_score, flat, method='Powell', 
                             args=(n, 0.02),
                             options={'maxiter': max_powell_iters // 2, 'xtol': 1e-10, 'ftol': 1e-12})
            pts2 = result2.x.reshape(n, 2)
            score2 = _compute_score(pts2)
            if score2 > current_score:
                current_score = score2
                current_pts = pts2.copy()
                flat = pts2.flatten()
        except Exception:
            pass
    
    # Stage 3: Exact -log(score) with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result3 = minimize(_neg_log_score, flat, method='Powell', 
                             args=(n,),
                             options={'maxiter': max_powell_iters, 'xtol': 1e-14, 'ftol': 1e-15})
            pts3 = result3.x.reshape(n, 2)
            score3 = _compute_score(pts3)
            if score3 > current_score:
                current_score = score3
                current_pts = pts3.copy()
        except Exception:
            pass
    
    return current_pts, current_score


def _simulated_annealing(points, deadline, seed=42, n_iter=3000):
    """Simulated annealing to escape local minima with adaptive step size."""
    n = 16
    rng = np.random.RandomState(seed)
    
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    # Compute scale of current configuration
    d = pdist(current)
    scale = d.max() if d.max() > 1e-10 else 1.0
    
    # Temperature schedule
    T_start = 0.5
    T_end = 1e-6
    cooling_rate = (T_end / T_start) ** (1.0 / n_iter)
    
    T = T_start
    base_step = 0.2 * scale
    
    for i in range(n_iter):
        if time.time() > deadline - 20:
            break
        
        # Adaptive step size: larger early, smaller late
        progress = i / n_iter
        step_size = base_step * (1.0 - 0.7 * progress)
        
        # Propose a move: perturb 1-3 points
        n_perturb = rng.randint(1, 4)
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
            if new_score > 0 and current_score > 0:
                delta = np.log(current_score / new_score)
            else:
                delta = 10.0
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


def _maximize_min_dist_constrained(points, max_dist, deadline, maxiter=200):
    """
    Maximize d_min subject to d_max <= max_dist using SLSQP.
    This is the direct geometric formulation that showed improvement in knowledge.
    """
    n = 16
    
    def objective(flat_pts):
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        # Penalty for constraint violation
        if d_max > max_dist:
            penalty = 1e4 * (d_max - max_dist)
        else:
            penalty = 0
        return -d_min + penalty
    
    def constraint_func(flat_pts):
        """Return max_dist - d_max (positive if satisfied)."""
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts)
        return max_dist - np.max(d)
    
    flat = points.flatten()
    
    try:
        result = minimize(
            objective,
            flat,
            method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': constraint_func}],
            options={'maxiter': maxiter, 'ftol': 1e-14, 'maxfun': 5000}
        )
        pts = result.x.reshape(n, 2)
        # Verify the result
        d_check = pdist(pts)
        if np.max(d_check) <= max_dist * 1.01:  # Small tolerance
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
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_points = None
    best_score = -1.0
    
    # Generate initial configurations
    configs = _generate_initial_configs(n=n, n_configs=8, seed=42)
    
    # Phase 1: SA + local refine for each config (budget: ~200s)
    for config_idx, config in enumerate(configs):
        if time.time() > deadline - 200:
            break
        
        # Simulated annealing with reduced iterations to save time
        sa_points, sa_score = _simulated_annealing(config, deadline - 150, seed=42 + config_idx, n_iter=2000)
        
        # Local refinement
        refined_points, refined_score = _local_refine(sa_points, deadline - 140, max_powell_iters=100)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
    
    # Phase 2: Direct geometric optimization - maximize d_min s.t. d_max <= threshold
    if best_points is not None and time.time() < deadline - 100:
        d = pdist(best_points)
        current_d_max = np.max(d)
        current_d_min = np.min(d)
        
        # Try several max_dist constraints to balance the distances
        # The optimal ratio has d_min/d_max maximized, so try constraining d_max
        # to values around current_d_max to force more uniform spacing
        max_dist_candidates = [
            current_d_max * 0.95,
            current_d_max * 0.98,
            current_d_max * 1.0,
            current_d_max * 1.02,
            current_d_max * 1.05,
        ]
        
        for max_dist in max_dist_candidates:
            if time.time() > deadline - 80:
                break
            
            # Maximize d_min subject to d_max <= max_dist
            opt_points = _maximize_min_dist_constrained(best_points, max_dist, deadline, maxiter=150)
            
            # Polish with local refinement
            refined, score = _local_refine(opt_points, deadline - 70, max_powell_iters=80)
            
            if score > best_score:
                best_score = score
                best_points = refined.copy()
    
    # Phase 3: Additional SA restarts from best point
    if best_points is not None and time.time() < deadline - 50:
        n_extra = 4
        for k in range(n_extra):
            if time.time() > deadline - 30:
                break
            
            # Perturb the best point with varying scales
            rng = np.random.RandomState(1000 + k)
            scale = pdist(best_points).max() * 0.15
            perturbed = best_points + rng.normal(0, scale, size=(n, 2))
            
            sa_points, sa_score = _simulated_annealing(perturbed, deadline - 25, seed=2000 + k, n_iter=1500)
            refined_points, refined_score = _local_refine(sa_points, deadline - 20, max_powell_iters=80)
            
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points.copy()
    
    # Phase 4: Final high-precision polish
    if best_points is not None and time.time() < deadline - 10:
        final_points, final_score = _local_refine(best_points, deadline, max_powell_iters=400)
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
