# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    if points is None or points.shape[0] < 2:
        return 0.0
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_objective(points, p=20.0):
    """
    Smooth surrogate using power means.
    Maximize (M_min / M_max)^2 where M_min approximates min distance and M_max approx max distance.
    """
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    
    # Clip to avoid numerical issues
    d = np.clip(d, 1e-10, 1e10)
    
    # Power mean for min: p -> -inf. Use negative p.
    # M_p = (sum(x^p)/N)^(1/p)
    # For min, we want p to be a large negative number.
    min_mean = (np.sum(d ** (-p)) / len(d)) ** (-1.0 / p)
    
    # Power mean for max: p -> +inf. Use positive p.
    max_mean = (np.sum(d ** p) / len(d)) ** (1.0 / p)
    
    if max_mean == 0:
        return 0.0
        
    ratio = min_mean / max_mean
    return ratio ** 2

def _logsumexp_surrogate(points, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for min/max distances.
    This provides better gradient information near the optimum.
    """
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    
    d = np.clip(d, 1e-10, 1e10)
    
    # Soft min: -logsumexp(-d/beta) / beta
    soft_min = -np.log(np.sum(np.exp(-d / beta))) * beta / len(d)
    
    # Soft max: logsumexp(d/beta) / beta
    soft_max = np.log(np.sum(np.exp(d / beta))) * beta / len(d)
    
    if soft_max == 0:
        return 0.0
        
    ratio = soft_min / soft_max
    return ratio ** 2

def _neg_smooth_objective(flat, p=20.0):
    points = flat.reshape(-1, 2)
    return -_smooth_objective(points, p)

def _neg_logsumexp_objective(flat, beta=50.0):
    points = flat.reshape(-1, 2)
    return -_logsumexp_surrogate(points, beta)

def _neg_true_objective(flat):
    points = flat.reshape(-1, 2)
    return -_compute_score(points)

def _refine(points, time_budget=10.0):
    """
    Refine points using multi-round polish with decreasing tolerance.
    Uses both smooth surrogates and true objective.
    """
    start = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    # Round 1: L-BFGS-B on log-sum-exp surrogate (coarse)
    if time.time() - start < time_budget * 0.25:
        try:
            res = minimize(_neg_logsumexp_objective, current_points.ravel(), 
                         method='L-BFGS-B', args=(30.0,),
                         options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_points = new_pts
                current_score = new_score
        except Exception:
            pass

    # Round 2: L-BFGS-B on power-mean surrogate (medium)
    if time.time() - start < time_budget * 0.45:
        try:
            res = minimize(_neg_smooth_objective, current_points.ravel(), 
                         method='L-BFGS-B', args=(30.0,),
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_points = new_pts
                current_score = new_score
        except Exception:
            pass

    # Round 3: Powell on true objective
    if time.time() - start < time_budget * 0.70:
        try:
            res = minimize(_neg_true_objective, current_points.ravel(), method='Powell',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_points = new_pts
                current_score = new_score
        except Exception:
            pass

    # Round 4: Nelder-Mead polish (fine)
    if time.time() - start < time_budget * 0.90:
        try:
            res = minimize(_neg_true_objective, current_points.ravel(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_points = new_pts
                current_score = new_score
        except Exception:
            pass

    # Round 5: Final ultra-fine Nelder-Mead
    if time.time() - start < time_budget * 0.98:
        try:
            res = minimize(_neg_true_objective, current_points.ravel(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_points = new_pts
                current_score = new_score
        except Exception:
            pass

    return current_points, current_score

def _generate_hexagonal_points(n=16):
    """Generate points in a hexagonal lattice pattern."""
    candidates = []
    q_max = 5
    for q in range(-q_max, q_max + 1):
        for r in range(-q_max, q_max + 1):
            if abs(q + r) <= q_max:
                x = 1.5 * q
                y = np.sqrt(3) * (r + 0.5 * q)
                candidates.append((x, y))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_indices = np.argsort(dists)
    selected = candidates[sorted_indices[:n]]
    return selected

def _generate_ring_points(radii, counts, angles_offsets=None):
    """Generate points on concentric rings."""
    points = []
    total = sum(counts)
    if total != 16:
        return None
    
    for i, (r, c) in enumerate(zip(radii, counts)):
        if c == 0:
            continue
        if c == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, c, endpoint=False)
            if angles_offsets and i < len(angles_offsets):
                angles += angles_offsets[i]
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points.extend(zip(x, y))
            
    return np.array(points)

def _simple_de(flat_obj, bounds, seed, maxiter, popsize, tol, start_time, deadline):
    """
    A simple, manual implementation of Differential Evolution to avoid
    potential overhead or issues with scipy's differential_evolution wrapper.
    """
    n_params = len(bounds)
    rng = np.random.default_rng(seed)
    
    # Initialize population
    pop = np.zeros((popsize, n_params))
    for i in range(popsize):
        for j in range(n_params):
            pop[i, j] = rng.uniform(bounds[j][0], bounds[j][1])
    
    # Evaluate initial population
    scores = np.array([flat_obj(pop[i]) for i in range(popsize)])
    
    best_idx = np.argmin(scores)
    best_x = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for gen in range(maxiter):
        if time.time() > deadline:
            break
        
        for i in range(popsize):
            if time.time() > deadline:
                break
            
            # Select three distinct random indices
            r1, r2, r3 = rng.choice(popsize, 3, replace=False)
            
            # Mutation
            F = 0.8
            C = 0.5
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Ensure bounds
            for j in range(n_params):
                if v[j] < bounds[j][0]:
                    v[j] = bounds[j][0]
                elif v[j] > bounds[j][1]:
                    v[j] = bounds[j][1]
            
            # Crossover
            j_rand = rng.integers(n_params)
            trial = pop[i].copy()
            for j in range(n_params):
                if rng.random() < C or j == j_rand:
                    trial[j] = v[j]
            
            # Evaluation
            trial_score = flat_obj(trial)
            
            # Selection
            if trial_score < scores[i]:
                scores[i] = trial_score
                pop[i] = trial
                if trial_score < best_score:
                    best_score = trial_score
                    best_x = trial.copy()
    
    return best_x, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    rng = np.random.default_rng(12345)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts, sc):
        nonlocal best_points, best_score
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()

    # 1. Initializations
    starts = []
    
    # 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    starts.append(grid_points)
    
    # Circle
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    circle_points = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(circle_points)
    
    # Hexagonal
    hex_points = _generate_hexagonal_points(16)
    starts.append(hex_points)
    
    # Rings: 1 center, 5 inner, 10 outer
    p_rings1 = _generate_ring_points([0.8, 1.8], [1, 5, 10], [0, 0, np.pi/10])
    if p_rings1 is not None:
        starts.append(p_rings1)
        
    # Rings: 1 center, 6 inner, 9 outer
    p_rings2 = _generate_ring_points([0.7, 1.6], [1, 6, 9], [0, 0, np.pi/9])
    if p_rings2 is not None:
        starts.append(p_rings2)
        
    # Random
    starts.append(rng.uniform(-2, 2, size=(16, 2)))
    starts.append(rng.uniform(-2, 2, size=(16, 2)))
    
    # Update best with initializations
    for s in starts:
        if s is not None and s.shape == (16, 2):
            sc = _compute_score(s)
            _update(s, sc)
    
    # 2. Global Search using Simple DE
    bounds = [(-3.0, 3.0)] * 32
    
    # Use log-sum-exp smooth objective for DE (better gradients)
    def de_obj(flat):
        return _neg_logsumexp_objective(flat, beta=30.0)

    try:
        # Run DE
        de_best_x, de_best_score = _simple_de(
            de_obj, 
            bounds, 
            seed=12345, 
            maxiter=120, 
            popsize=24, 
            tol=1e-10,
            start_time=start_time,
            deadline=deadline
        )
        
        candidate_points = de_best_x.reshape(-1, 2)
        candidate_score = _compute_score(candidate_points)
        _update(candidate_points, candidate_score)
        
        # Add to starts for refinement
        starts.append(candidate_points)
        
    except Exception:
        pass
    
    # 3. Local Refinement Loop
    
    if best_points is not None:
        starts.insert(0, best_points)
        
    # Remove duplicates
    unique_starts = []
    for s in starts:
        if s is None or s.shape != (16, 2):
            continue
        is_dup = False
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)
            
    # Prioritize the current best
    if best_points is not None:
        time_left = deadline - time.time()
        if time_left > 30:
            refined_pts, refined_score = _refine(best_points, time_budget=min(100, time_left - 10))
            _update(refined_pts, refined_score)
            
    # Refine other starts
    for idx, start_pts in enumerate(unique_starts[1:]):
        time_left = deadline - time.time()
        if time_left < 15:
            break
            
        num_remaining = len(unique_starts) - 1 - idx
        per_start_time = (time_left - 5) / max(1, num_remaining)
        per_start_time = min(60, per_start_time)
        
        if per_start_time < 5:
            continue
            
        try:
            refined_pts, refined_score = _refine(start_pts, time_budget=per_start_time)
            _update(refined_pts, refined_score)
        except Exception:
            pass

    # 4. Final Polish
    if best_points is not None:
        time_left = deadline - time.time()
        if time_left > 10:
            final_pts, final_score = _refine(best_points, time_budget=time_left - 2)
            _update(final_pts, final_score)

    if best_points is None:
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    if not np.all(np.isfinite(best_points)):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    return best_points

# EVOLVE-BLOCK-END
