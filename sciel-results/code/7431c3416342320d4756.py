"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _bary_to_cart(bary):
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _softmin_objective(x, beta=80.0):
    """Smooth surrogate: negative soft-min of triangle areas.
    soft_min = min_a - (1/beta)*log(sum(exp(-beta*(a - min_a))))
    This is differentiable and approaches true min as beta -> inf."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    shifted = areas - min_area
    # Numerically stable log-sum-exp
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_area - log_sum / beta
    return -soft_min

def _hard_objective(x):
    """Hard objective: negative min triangle area."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _score_bary(bary):
    points = _bary_to_cart(bary)
    return _min_triangle_area_vectorized(points)

def _structured_init_bary(seed=42):
    """Generate structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    patterns = []
    
    # Pattern A: Symmetric with vertices, edge midpoints, centroid
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
    ])
    patterns.append(pA)
    
    # Pattern B: Hexagonal-ish
    pB = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.33, 0.33],
        [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
    ])
    patterns.append(pB)
    
    # Pattern C: More spread
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Random structured
    pD = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [rng.uniform(0.3, 0.7), 0.0],
        [rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)],
        [0.0, rng.uniform(0.3, 0.7)],
        [1/3, 1/3],
        [rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.1, 0.3), rng.uniform(0.7, 0.9)],
        [rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.5)]
    ])
    patterns.append(pD)
    
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    jitter = rng.normal(0, 0.02, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    return bary

def _random_init_bary(seed):
    """Random initialization in barycentric coords."""
    rng = np.random.RandomState(seed)
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                bary[i] = [u, v]
                break
    return bary

def _l_bfgs_softmin(bary0, maxiter=400, beta=80.0):
    """L-BFGS-B optimization using soft-min surrogate for smooth gradients."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        def obj(x):
            return _softmin_objective(x, beta=beta)
        
        res = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _l_bfgs_hard(bary0, maxiter=200):
    """L-BFGS-B optimization using hard min (for final polish)."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _hard_objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _active_set_refine(bary0, maxiter=100):
    """Refine by identifying active (binding) triples and optimizing to equalize them.
    
    Strategy: find the triples achieving the minimum area, then use a penalty
    method that strongly penalizes any area below the current minimum while
    encouraging the minimum to increase.
    """
    try:
        points = _bary_to_cart(bary0)
        areas = _all_triangle_areas(points)
        min_area = np.min(areas)
        
        # Identify active triples (within tolerance of minimum)
        tol = 0.05 * min_area if min_area > 0 else 1e-6
        active_mask = areas <= min_area + tol
        active_count = np.sum(active_mask)
        
        if active_count < 2:
            return bary0.copy()
        
        # Penalty objective: maximize min area with soft constraint on active triples
        # Use a barrier-like approach: penalize deviations of active areas from target
        target = min_area
        
        def penalty_obj(x):
            bary = x.reshape(N, 2)
            bary = _clamp_bary(bary)
            pts = _bary_to_cart(bary)
            a = _all_triangle_areas(pts)
            min_a = np.min(a)
            # Penalty for active triples falling below target
            active_dev = np.maximum(0, target - a[active_mask])
            penalty = 100.0 * np.sum(active_dev ** 2)
            # Also penalize any area below a fraction of target
            all_dev = np.maximum(0, 0.9 * target - a)
            penalty += 50.0 * np.sum(all_dev ** 2)
            return -min_a + penalty
        
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        res = minimize(
            penalty_obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        
        # Verify improvement
        score_opt = _score_bary(bary_opt)
        score_orig = _score_bary(bary0)
        if score_opt > score_orig:
            return bary_opt
        return bary0.copy()
    except Exception:
        return bary0.copy()

def _simulated_annealing(seed=42, time_limit=60.0, T_start=0.05, T_end=0.001, cooling=0.9995):
    """Simulated Annealing for global search."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Initial random point
    bary = np.zeros((N, 2))
    for i in range(N):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1 - u)
        bary[i] = [u, v]
    
    best_bary = bary.copy()
    best_score = _score_bary(bary)
    current_score = best_score
    T = T_start
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
        
        # Generate new candidate
        # Step size adapts to temperature
        step_size = 0.05 * T / T_start + 0.01
        bary_new = bary + rng.normal(0, step_size, size=bary.shape)
        bary_new = _clamp_bary(bary_new)
        
        score_new = _score_bary(bary_new)
        delta = score_new - current_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.uniform() < np.exp(delta / T):
            bary = bary_new
            current_score = score_new
            
            if current_score > best_score:
                best_score = current_score
                best_bary = bary.copy()
        
        T *= cooling
        
    return best_bary, best_score

def _nelder_mead_optimize(bary0, maxiter=200):
    """Nelder-Mead for local refinement."""
    try:
        x0 = bary0.flatten()
        res = minimize(
            _hard_objective, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    # Phase 1: Multiple Simulated Annealing runs
    # SA is good for global exploration in non-convex spaces
    sa_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for seed in sa_seeds:
        if time_left() < 15.0:
            break
        
        # Allocate time for SA based on remaining budget
        # Leave at least 10s for polishing
        sa_time = min(10.0, max(2.0, time_left() - 10.0))
        
        bary_sa, score_sa = _simulated_annealing(seed=seed, time_limit=sa_time)
        
        if score_sa > best_score:
            best_score = score_sa
            best_bary = bary_sa.copy()
            
            # Polish immediately with Nelder-Mead if time permits
            if time_left() > 5.0:
                bary_nm = _nelder_mead_optimize(bary_sa, maxiter=200)
                score_nm = _score_bary(bary_nm)
                if score_nm > best_score:
                    best_score = score_nm
                    best_bary = bary_nm.copy()

    # Phase 2: Final refinement of the best solution found
    if best_bary is not None and time_left() > 5.0:
        # Run Nelder-Mead again with higher precision
        bary_final = _nelder_mead_optimize(best_bary, maxiter=300)
        score_final = _score_bary(bary_final)
        if score_final > best_score:
            best_score = score_final
            best_bary = bary_final.copy()
            
        # One last small perturbation + polish
        if time_left() > 3.0:
            rng = np.random.RandomState(999)
            pert = rng.normal(0, 0.01, size=(N, 2))
            bary_pert = _clamp_bary(best_bary + pert)
            bary_pert_opt = _nelder_mead_optimize(bary_pert, maxiter=100)
            score_pert = _score_bary(bary_pert_opt)
            if score_pert > best_score:
                best_score = score_pert
                best_bary = bary_pert_opt.copy()

    # Fallback if SA failed completely
    if best_bary is None:
        best_bary = _structured_init_bary(seed=42)
        # Polish the fallback
        if time_left() > 5.0:
            best_bary = _nelder_mead_optimize(best_bary, maxiter=200)

    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
