"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for efficiency
# i < j < k
_idx_i, _idx_j, _idx_k = np.indices((N, N, N)).reshape(3, -1)
_mask = (_idx_i < _idx_j) & (_idx_j < _idx_k)
_TRI_I = _idx_i[_mask]
_TRI_J = _idx_j[_mask]
_TRI_K = _idx_k[_mask]

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _smooth_min_area(pts, beta):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    """
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    if min_a < 1e-10:
         return 0.0
         
    # Shift for numerical stability
    y = -beta * (areas - min_a)
    sum_exp = np.sum(np.exp(y))
    if sum_exp == 0:
        return min_a
    log_sum_exp = np.log(sum_exp)
    
    soft_min = min_a - (1.0 / beta) * log_sum_exp
    return soft_min

def _objective(params, beta):
    """
    Objective for differential evolution.
    params: flattened 2D points.
    We normalize the points to unit area (or similar scale) internally to help the optimizer?
    Actually, scaling invariance means we can fix one point or normalize.
    Let's normalize so that the hull area is 1.0.
    """
    pts = params.reshape(N, 2)
    
    # Normalize to unit hull area
    h_area = _hull_area(pts)
    if h_area < 1e-9:
        return 1e9
        
    pts_norm = pts / np.sqrt(h_area)
    
    s_min = _smooth_min_area(pts_norm, beta)
    if s_min <= 0:
        return 1e9
        
    # Maximize s_min (since hull area is 1)
    return -s_min

def _exact_objective(params):
    """
    Objective for local refinement (Nelder-Mead or Powell).
    """
    pts = params.reshape(N, 2)
    s = _compute_score(pts)
    if s == 0.0:
        return 1e9
    return -s

def _generate_initial_params(rng):
    """
    Generate a random set of points.
    To help the optimizer, we can start with points in a convex position, e.g., on a circle or random convex polygon.
    """
    # Random convex polygon: sort angles, random radii
    angles = np.sort(rng.uniform(0, 2 * np.pi, N))
    radii = rng.uniform(0.5, 1.0, N)
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return pts.flatten()

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Global Search with Differential Evolution
    # Bounds: points in [-1, 1] x [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # We want to maximize min_area / hull_area.
    # DE minimizes. So we minimize -score.
    
    # Use a moderate population size and max iterations.
    # popsize=15, maxiter=200 might be too slow?
    # Let's check time budget.
    # Each eval: _hull_area + _smooth_min_area.
    # _smooth_min_area is vectorized over C(13,3)=286 triangles. Fast.
    # _hull_area is O(N log N). Fast.
    # So DE should be fast.
    
    # Strategy:
    # 1. Run DE for a few seconds.
    # 2. Take the best result and refine with Nelder-Mead.
    
    # Let's run DE in a loop, checking time.
    # Or just run DE with a timeout? SciPy DE doesn't have a direct timeout, but we can wrap it.
    # Simpler: Run DE with limited iterations, then refine.
    
    # Initial population
    init_pop = []
    for _ in range(10):
        init_pop.append(_generate_initial_params(rng))
    
    try:
        # differential_evolution doesn't take an initial population directly in all versions?
        # scipy.optimize.differential_evolution takes `x0` as a single point or None.
        # It generates its own population.
        # We can use `seed=rng` to control it.
        
        # Let's use a smaller popsize for speed, e.g., 10.
        # maxiter=100.
        
        res_de = differential_evolution(
            _objective,
            bounds,
            args=(50.0,), # beta for smooth min
            maxiter=150,
            popsize=15,
            tol=1e-8,
            seed=rng,
            init='latinhypercube',
            polish=False # We will polish ourselves
        )
        
        if res_de.fun < 1e8:
            de_pts = res_de.x.reshape(N, 2)
            de_score = _compute_score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
                
    except Exception:
        pass
    
    # Check time
    if (time.time() - start_time) > deadline - 10.0:
        if best_pts is not None:
            return best_pts
        else:
            # Fallback
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            return np.column_stack((np.cos(angles), np.sin(angles)))
            
    # Phase 2: Local Refinement
    # Use the best point from DE (or a random one if DE failed)
    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _compute_score(best_pts)
        
    # Try multiple starts for refinement to escape local minima
    # Use the DE best, and a few random perturbations of it.
    
    starts = [best_pts.flatten()]
    for _ in range(5):
        # Perturb best_pts
        perturb = rng.normal(0, 0.1, (N, 2))
        p = best_pts + perturb
        starts.append(p.flatten())
        
    # Also add regular polygon and a few random convex polygons
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    starts.append(np.column_stack((np.cos(angles_reg), np.sin(angles_reg))).flatten())
    
    for start in starts:
        if (time.time() - start_time) > deadline - 5.0:
            break
            
        try:
            # Nelder-Mead is robust for low-dimensional problems (26D here)
            res_nm = minimize(
                _exact_objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.fun < 1e8:
                nm_pts = res_nm.x.reshape(N, 2)
                nm_score = _compute_score(nm_pts)
                
                if nm_score > best_score:
                    best_score = nm_score
                    best_pts = nm_pts.copy()
                    
        except Exception:
            pass
            
    # Phase 3: Final Polish with higher precision if time remains
    # Use Powell or COBYLA? Nelder-Mead is usually enough.
    # Let's try one more round of Nelder-Mead with tighter tolerances on the best point.
    
    if (time.time() - start_time) < deadline - 2.0 and best_pts is not None:
        try:
            res_polish = minimize(
                _exact_objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_polish.fun < 1e8:
                polish_pts = res_polish.x.reshape(N, 2)
                polish_score = _compute_score(polish_pts)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_pts = polish_pts.copy()
        except Exception:
            pass
            
    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_pts

# EVOLVE-BLOCK-END
