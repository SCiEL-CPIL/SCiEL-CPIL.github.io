import numpy as np
from scipy.spatial.distance import pdist
import time
import scipy.optimize as so
from scipy.optimize import LinearConstraint, NonlinearConstraint

def _compute_score(points):
    """Compute R = (d_min / d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _pair_indices(n):
    """Return indices for all pairs (i, j) with i < j."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return np.array(pairs)

def _soft_max(vals, beta=100.0):
    """Compute soft-max of squared distances using log-sum-exp for smoothness."""
    max_val = np.max(vals)
    if max_val < 1e-15:
        return 0.0
    exp_vals = np.exp(-beta * (vals - max_val))
    log_sum = np.log(np.sum(exp_vals))
    return max_val + log_sum / beta

def _soft_min(vals, beta=100.0):
    """Compute soft-min of squared distances using log-sum-exp for smoothness."""
    min_val = np.min(vals)
    if min_val < 1e-15:
        return 0.0
    exp_vals = np.exp(beta * (vals - min_val))
    log_sum = np.log(np.sum(exp_vals))
    return min_val - log_sum / beta

def _epigraph_optimize(points, max_iter=300, tol=1e-9, beta=50.0):
    """
    Epigraph formulation: maximize t subject to:
    - d_sq[i,j] >= t for all pairs (min distance constraint)
    - d_sq[i,j] <= 1 for all pairs (max distance constraint)
    
    Uses smooth approximations with proper gradients.
    Variable: x = [point coords..., t]
    """
    n = len(points)
    pairs = _pair_indices(n)
    m = len(pairs)
    x0 = np.concatenate([points.flatten(), [0.1]])  # Initial t guess
    
    def objective(v):
        """Minimize -t (to maximize t)"""
        return -v[-1]
    
    def objective_grad(v):
        grad = np.zeros(2 * n + 1)
        grad[-1] = -1.0
        return grad
    
    # Smooth constraint: soft_min(d_sq) - t >= 0
    def min_dist_constraint(v):
        pts = v[:2*n].reshape(n, 2)
        t = v[-1]
        d_sq = pdist(pts) ** 2
        return _soft_min(d_sq, beta=beta) - t
    
    def min_dist_jac(v):
        pts = v[:2*n].reshape(n, 2)
        t = v[-1]
        d_sq = pdist(pts) ** 2
        min_val = np.min(d_sq)
        exp_vals = np.exp(beta * (d_sq - min_val))
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros(2 * n + 1)
        for k, (i, j) in enumerate(pairs):
            w = weights[k]
            diff = pts[i] - pts[j]
            # d(soft_min)/d(pts[i]) = sum_k w_k * 2*(pts[i_k] - pts[j_k])
            grad[2*i:2*i+2] += w * 2 * diff
            grad[2*j:2*j+2] -= w * 2 * diff
        grad[-1] = -1.0
        return grad
    
    # Smooth constraint: 1 - soft_max(d_sq) >= 0
    def max_dist_constraint(v):
        pts = v[:2*n].reshape(n, 2)
        d_sq = pdist(pts) ** 2
        return 1.0 - _soft_max(d_sq, beta=beta)
    
    def max_dist_jac(v):
        pts = v[:2*n].reshape(n, 2)
        d_sq = pdist(pts) ** 2
        max_val = np.max(d_sq)
        exp_vals = np.exp(-beta * (d_sq - max_val))
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros(2 * n + 1)
        for k, (i, j) in enumerate(pairs):
            w = weights[k]
            diff = pts[i] - pts[j]
            # d(soft_max)/d(pts[i]) = sum_k w_k * 2*(pts[i_k] - pts[j_k])
            grad[2*i:2*i+2] -= w * 2 * diff
            grad[2*j:2*j+2] += w * 2 * diff
        return grad
    
    cons_min = NonlinearConstraint(min_dist_constraint, 0, np.inf, jac=min_dist_jac)
    cons_max = NonlinearConstraint(max_dist_constraint, 0, np.inf, jac=max_dist_jac)
    
    bounds = [(-5, 5)] * (2 * n) + [(0.01, 1.0)]
    
    try:
        res = so.minimize(objective, x0, method='trust-constr', 
                         jac=objective_grad,
                         bounds=bounds, 
                         constraints=[cons_min, cons_max],
                         options={'maxiter': max_iter, 'gtol': tol, 'xtol': tol})
        if res.x is not None:
            return res.x[:2*n].reshape(n, 2)
    except Exception:
        pass
    return points

def _local_search(points, time_budget=60.0, seed=42):
    """
    Fast local search using gradient-based optimization with epigraph formulation.
    Multiple restarts with small perturbations.
    """
    n = len(points)
    rng = np.random.RandomState(seed)
    
    # Normalize
    d = pdist(points)
    d_max = d.max()
    if d_max > 1e-9:
        current = points / d_max
    else:
        current = points.copy()
    
    best_points = current.copy()
    best_score = _compute_score(best_points)
    
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < time_budget:
        attempts += 1
        if attempts > 50:
            break
        
        # Perturb current best
        if attempts == 1:
            x0 = current.copy()
        else:
            x0 = best_points + rng.randn(n, 2) * 0.02
        
        try:
            result = _epigraph_optimize(x0, max_iter=100, tol=1e-8, beta=30.0)
            d_new = pdist(result)
            d_min_new = d_new.min()
            d_max_new = d_new.max()
            
            if d_max_new > 1e-9 and d_min_new > 1e-9:
                score_new = (d_min_new / d_max_new) ** 2
                if score_new > best_score:
                    best_score = score_new
                    best_points = result.copy()
                    current = result.copy()
        except Exception:
            pass
    
    return best_points, best_score

def _generate_initial_points(seed=0):
    """Generate multiple structured initial configurations."""
    n = 16
    rng = np.random.RandomState(seed)
    candidates = []

    # 1. Regular polygon + center (8-gon + inner 8-gon scaled)
    angles_outer = 2 * np.pi * np.arange(8) / 8
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    candidates.append(np.vstack([outer, inner]))

    # 2. 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.column_stack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2)
    candidates.append(grid)

    # 3. Two concentric 8-gons
    angles1 = 2 * np.pi * np.arange(8) / 8
    angles2 = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    r1 = 1.0
    r2 = 0.6
    p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    candidates.append(np.vstack([p1, p2]))

    # 4. Hexagonal-ish arrangement: 1 center + 6 ring + 9 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(6) / 6
    ring1 = 0.8 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = 2 * np.pi * np.arange(9) / 9
    ring2 = 1.5 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 5. Random
    candidates.append(rng.randn(n, 2) * 0.5)

    # 6. Another structured: 2x8 rectangle with jitter
    xs = np.tile(np.array([-1, 1]), 8)
    ys = np.linspace(-1, 1, 16)
    rect = np.column_stack([xs, ys])
    candidates.append(rect + rng.randn(16, 2) * 0.05)

    # 7. Random with larger spread
    candidates.append(rng.randn(n, 2) * 2.0)

    # 8. 3x3 grid + 1 center (9 + 1 = 10, need 6 more)
    xs3 = np.linspace(-1, 1, 3)
    ys3 = np.linspace(-1, 1, 3)
    grid3 = np.column_stack(np.meshgrid(xs3, ys3, indexing='ij')).reshape(-1, 2)
    extra_angles = 2 * np.pi * np.arange(7) / 7
    extra = 1.5 * np.column_stack([np.cos(extra_angles), np.sin(extra_angles)])
    candidates.append(np.vstack([grid3, extra[:7]]))

    # 9. Two squares (4x4)
    angles_sq = 2 * np.pi * np.arange(4) / 4
    sq1 = 1.0 * np.column_stack([np.cos(angles_sq), np.sin(angles_sq)])
    sq2 = 0.6 * np.column_stack([np.cos(angles_sq + np.pi/4), np.sin(angles_sq + np.pi/4)])
    candidates.append(np.vstack([sq1, sq2, sq1 * 1.5, sq2 * 1.5]))

    return candidates

def _polish(points, time_budget=50.0, seed=7):
    """
    Final polish using epigraph optimization with tighter tolerances.
    """
    n = len(points)
    d = pdist(points)
    d_max = d.max()
    if d_max > 1e-9:
        normalized = points / d_max
    else:
        normalized = points.copy()

    start_time = time.time()
    best = normalized.copy()
    best_score = _compute_score(best)
    rng = np.random.RandomState(seed)

    for attempt in range(12):
        if time.time() - start_time > time_budget:
            break

        if attempt == 0:
            x0 = normalized.copy()
        else:
            x0 = best + rng.randn(n, 2) * 0.005

        try:
            result = _epigraph_optimize(x0, max_iter=200, tol=1e-10, beta=40.0)
            d_new = pdist(result)
            d_min_new = d_new.min()
            d_max_new = d_new.max()

            if d_max_new > 1e-9 and d_min_new > 1e-9:
                new_score = (d_min_new / d_max_new) ** 2
                if new_score > best_score:
                    best_score = new_score
                    best = result.copy()
        except Exception:
            pass

    return best, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 610.0  # Safety margin before 630s kill

    # Generate initial candidates
    initial_points_list = _generate_initial_points(seed=42)

    best_points = initial_points_list[0].copy()
    best_score = _compute_score(best_points)

    # Phase 1: Optimization from each initial configuration using local search
    for i, init_pts in enumerate(initial_points_list):
        if time.time() > deadline - 100.0:
            break

        time_budget = min(30.0, deadline - time.time() - 80.0)
        if time_budget < 5.0:
            break

        try:
            result_pts, result_score = _local_search(init_pts, time_budget=time_budget, seed=42+i)
            if result_score > best_score:
                best_points = result_pts
                best_score = result_score
        except Exception:
            pass

    # Phase 2: Final polish with tighter tolerances
    remaining = deadline - time.time() - 10.0
    if remaining > 20.0:
        try:
            polished_pts, polished_score = _polish(best_points, time_budget=remaining - 10.0, seed=7)
            if polished_score > best_score:
                best_points = polished_pts
                best_score = polished_score
        except Exception:
            pass

    # Ensure valid output
    d = pdist(best_points)
    d_max = d.max()
    if d_max > 1e-9:
        best_points = best_points / d_max

    return best_points
