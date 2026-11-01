import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def soft_min_score(x, tau):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    This is differentiable and approximates -min(A) as tau -> 0.
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 1e10
    except Exception:
        return 1e10
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
        
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def make_ring_config_fn(splits):
    """
    Returns a function that maps a parameter vector to points.
    Parameter vector: [r_0, ..., r_{k-1}, theta_0, ..., theta_{k-1}]
    """
    k = len(splits)
    def _config(params):
        points = []
        for i, n_i in enumerate(splits):
            if n_i == 0:
                continue
            r_i = params[i]
            th_i = params[k + i]
            angles = th_i + np.linspace(0, 2*np.pi, n_i, endpoint=False)
            pts = np.column_stack((r_i * np.cos(angles), r_i * np.sin(angles)))
            points.append(pts)
        
        if not points:
            return np.zeros((N, 2))
            
        result = np.vstack(points)
        
        # Ensure exactly N points (should be true if splits sum to N)
        if len(result) < N:
            extra = np.zeros((N - len(result), 2))
            result = np.vstack([result, extra])
        elif len(result) > N:
            result = result[:N]
            
        return result
    return _config

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # Phase 0: Global Search over Ring Parameterizations
    # We explore a few common symmetric split patterns.
    splits_to_try = [
        [7, 7],
        [5, 4, 5],
        [4, 4, 3, 3],
        [6, 4, 4],
        [5, 5, 4],
        [4, 5, 5]
    ]
    
    candidate_starts = []
    
    for splits in splits_to_try:
        if time.time() - start_time > deadline - 60:
            break
            
        k = len(splits)
        # Parameters: k radii, k angles
        bounds = [(0.1, 1.5)] * k + [(0, 2*np.pi)] * k
        
        config_fn = make_ring_config_fn(splits)
        
        def neg_ring_score(params):
            try:
                pts = config_fn(params)
                return -compute_score(pts)
            except Exception:
                return 1e10

        # Use Differential Evolution to find good ring parameters
        try:
            res = differential_evolution(
                neg_ring_score, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-5,
                update_interval=5
            )
            best_params = res.x
            pts = config_fn(best_params)
            try_update(pts)
            candidate_starts.append(pts)
        except Exception:
            pass

    # Ensure we have at least one starting point (fallback to regular 14-gon if DE failed)
    if not candidate_starts:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        fallback = np.column_stack((np.cos(angles), np.sin(angles)))
        try_update(fallback)
        candidate_starts.append(fallback)

    # Phase 1: Full-space Refinement from Best Candidates
    # We take the top candidates from Phase 0 and refine them in the full 28D space.
    
    # Sort candidates by score to prioritize best ones
    candidate_starts.sort(key=lambda p: compute_score(p), reverse=True)
    
    # Limit number of refinements to save time
    max_refinements = 6
    if time.time() - start_time < deadline - 30:
        for i in range(min(max_refinements, len(candidate_starts))):
            if time.time() - start_time > deadline - 15:
                break
                
            start_pt = candidate_starts[i]
            x0 = start_pt.flatten()
            
            # Annealing schedule for tau
            # Start with larger tau for smooth landscape, decrease for sharpness
            taus = [0.1, 0.05, 0.01, 0.005, 0.001]
            
            current_x = x0
            for tau in taus:
                if time.time() - start_time > deadline - 10:
                    break
                
                try:
                    res = minimize(
                        soft_min_score,
                        current_x,
                        args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    current_x = res.x
                    new_p = current_x.reshape((N, 2))
                    try_update(new_p)
                except Exception:
                    pass

    # Phase 2: Basin Hopping from Global Best
    if best_points is not None and time.time() - start_time < deadline - 15:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9},
                minsteps=2, T=0.5, stepsize=0.2,
                interval=5, disp=False
            )
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass
    
    # Phase 3: Final Powell/Nelder-Mead Polish
    if best_points is not None and time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
            
    return best_points
