import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _softmin_score(points, alpha):
    """
    Smooth approximation of the score using softmin of triangle areas.
    Returns: softmin_val / hull_area
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -1/alpha * log(sum(exp(-alpha * areas)))
    # Subtract max for numerical stability (which is min for softmin logic usually, 
    # but here we want to approximate min, so we look at the smallest values.
    # Actually, standard softmin is -1/alpha * log(sum(exp(-alpha * x))).
    # If x are areas, and we want min, this works.
    # To prevent overflow in exp(-alpha * x), we should subtract the minimum?
    # Let's check: if x_min is very small, -alpha*x_min is large positive? No.
    # exp(-alpha * x). If x is small, -alpha*x is large negative? No.
    # If x is small, alpha*x is small. -alpha*x is close to 0.
    # If x is large, alpha*x is large. -alpha*x is large negative.
    # So the term with the smallest x has the largest exponent.
    # To stabilize, we can subtract the minimum area from all areas before exp?
    # Let m = min(areas).
    # sum(exp(-alpha * (areas - m))) = sum(exp(-alpha*areas + alpha*m)) = exp(alpha*m) * sum(exp(-alpha*areas))
    # log(sum(exp(-alpha*areas))) = alpha*m + log(sum(exp(-alpha*(areas-m))))
    # softmin = -1/alpha * (alpha*m + log(...)) = -m - 1/alpha * log(...)
    # Wait, softmin should be close to m.
    # If we use the formula directly:
    # Let's just use the stable version.
    
    a_min = np.min(areas)
    # We want to compute -1/alpha * log( sum( exp(-alpha * areas) ) )
    # = -1/alpha * ( alpha * a_min + log( sum( exp(-alpha * (areas - a_min)) ) ) )
    # = -a_min - 1/alpha * log( sum( exp(-alpha * (areas - a_min)) ) )
    
    shifted = areas - a_min
    # exp(-alpha * shifted) is safe because shifted >= 0, so exponent <= 0
    exp_terms = np.exp(-alpha * shifted)
    log_sum = np.log(np.sum(exp_terms))
    
    softmin_val = -a_min - (1.0 / alpha) * log_sum
    # Note: softmin_val <= a_min. It is a lower bound on the min.
    # We want to MAXIMIZE this.
    
    return softmin_val / hull_area

def _neg_softmin_score(params, alpha):
    """Objective for softmin-based optimization."""
    points = params.reshape(N, 2)
    s = _softmin_score(points, alpha)
    return -s

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _simulate_annealing(start_pts, deadline, rng, initial_temp=0.5, final_temp=1e-6, 
                         initial_alpha=5.0, final_alpha=100.0, 
                         steps_per_temp=50, cooling_rate=0.98):
    """
    Simulated Annealing with annealing alpha for softmin.
    """
    current_pts = start_pts.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    temp = initial_temp
    alpha = initial_alpha
    
    # We will iterate until deadline or temp is too low
    n_iter = 0
    total_iters = 2000 # safety cap
    
    while time.time() < deadline - 5 and temp > final_temp and n_iter < total_iters:
        # Anneal alpha and temp
        # We want to increase alpha (sharper min) and decrease temp (more greedy)
        # But we need to balance.
        # Let's just step through a fixed schedule of alphas and temps
        # Or adaptively.
        
        # Generate a perturbation
        # Step size should be related to temp
        step_size = temp * 0.1 # Scale down
        if step_size < 1e-5:
            step_size = 1e-5
            
        trial_pts = current_pts + rng.normal(0, step_size, (N, 2))
        
        trial_score = _score(trial_pts)
        
        # Acceptance probability
        if trial_score > current_score:
            current_pts = trial_pts
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Probability of accepting worse move
            # delta_score = current_score - trial_score (positive)
            delta = current_score - trial_score
            # p = exp(-delta / temp)
            # To avoid overflow/underflow issues with very small temp, 
            # if temp is tiny, p is 0.
            if temp > 1e-8:
                p = np.exp(-delta / temp)
                if rng.uniform() < p:
                    current_pts = trial_pts
                    current_score = trial_score
        
        # Update parameters
        temp *= cooling_rate
        # Increase alpha slowly to focus on the min
        alpha *= 1.01 
        
        n_iter += 1
        
    return best_pts, best_score

def _polish(points, deadline, rng):
    """
    Final polishing using Nelder-Mead on the hard score.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    # Nelder-Mead on hard score
    if time.time() < deadline - 2:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
            
    # One more pass with L-BFGS-B on softmin with high alpha
    if time.time() < deadline - 1:
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(200.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass

    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Perturbed regular polygons
    for perturb in [0.01, 0.05, 0.1, 0.2]:
        angles_p = 2 * np.pi * np.arange(N) / N + rng.uniform(-perturb, perturb, N)
        radii_p = 1.0 + rng.uniform(-perturb, perturb, N)
        initial_configs.append(np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)]))
        
    # 3. Random points in unit circle
    for _ in range(5):
        angles_r = rng.uniform(0, 2*np.pi, N)
        radii_r = rng.uniform(0.5, 1.0, N)
        initial_configs.append(np.column_stack([radii_r * np.cos(angles_r), radii_r * np.sin(angles_r)]))
        
    # 4. Two-ring structure (inner and outer)
    # 7 outer, 6 inner? Or 6 outer, 7 inner?
    # Let's try 7 outer, 6 inner
    n_outer = 7
    n_inner = 6
    angles_o = 2 * np.pi * np.arange(n_outer) / n_outer
    angles_i = 2 * np.pi * np.arange(n_inner) / n_inner + np.pi/n_inner # stagger
    pts_o = np.column_stack([np.cos(angles_o), np.sin(angles_o)])
    pts_i = np.column_stack([0.5 * np.cos(angles_i), 0.5 * np.sin(angles_i)])
    initial_configs.append(np.vstack([pts_o, pts_i]))
    
    # 5. Ellipse based
    for a in [1.2, 1.5]:
        for b in [0.8, 0.5]:
            angles_e = 2 * np.pi * np.arange(N) / N
            initial_configs.append(np.column_stack([a * np.cos(angles_e), b * np.sin(angles_e)]))

    # Sort initial configs by score to prioritize good starts
    scored_configs = []
    for config in initial_configs:
        try:
            s = _score(config)
            scored_configs.append((s, config))
        except Exception:
            pass
    scored_configs.sort(key=lambda x: -x[0])
    
    # Allocate time for SA runs
    # We have ~145s. 
    # Let's run SA on the top 5-10 configurations.
    # Each SA run should take ~10-20s.
    
    n_configs_to_try = min(len(scored_configs), 8)
    time_per_config = (deadline - start_time - 10) / n_configs_to_try # reserve 10s for polish
    
    for i in range(n_configs_to_try):
        if time.time() > deadline - 10:
            break
            
        s_init, config = scored_configs[i]
        
        # Run Simulated Annealing
        # Adjust parameters based on time left
        # If it's the first config, give it more time
        alloc_time = time_per_config if i < n_configs_to_try - 1 else (deadline - time.time() - 5)
        
        try:
            sa_pts, sa_score = _simulate_annealing(
                config, 
                start_time + alloc_time, 
                rng,
                initial_temp=0.5, 
                final_temp=1e-6,
                cooling_rate=0.99
            )
            
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts
        except Exception:
            pass
            
    # Polish the best result found
    if best_points is not None:
        # Use remaining time for polishing
        polish_deadline = deadline - 2
        polished_pts, polished_score = _polish(best_points, polish_deadline, rng)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
            
    # Fallback
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
