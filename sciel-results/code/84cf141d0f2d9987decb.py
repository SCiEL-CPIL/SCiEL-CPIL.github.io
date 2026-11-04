"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices for fast area calculation
_triple_indices = np.array(list(combinations(range(N), 3)))

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _params_to_points(params):
    """Convert parameter vector (22 values) to point array (13x2)."""
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    return pts

def _points_to_params(pts):
    """Convert point array to parameter vector."""
    return pts[2:].flatten()

def _normalize_pts(pts):
    """
    Normalize points so that p[0] is at (0,0) and p[1] is at (1,0).
    This removes the 3 degrees of freedom (translation, rotation, scale) 
    by fixing the first two points.
    """
    if np.linalg.norm(pts[1] - pts[0]) < 1e-9:
        return None
    
    pts = pts.copy()
    pts = pts - pts[0]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return None
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    return pts

def _objective(params):
    """
    Objective function for minimization: -min_ratio.
    Returns a large positive number if invalid.
    """
    pts = _params_to_points(params)
    score = _compute_min_ratio(pts)
    if score <= 0:
        return 1e9
    return -score

def _generate_regular_polygon_params():
    """
    Generate parameters for a regular 13-gon centered at (0.5, 0.5) roughly, 
    then normalized.
    Actually, for Heilbronn, points are often near the boundary or in rings.
    A regular polygon is a good baseline.
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    pts = np.column_stack((radii * np.cos(angles) + 0.5, radii * np.sin(angles) + 0.5))
    
    # Normalize: pick two points to fix. 
    # For a regular polygon, p[0] and p[1] are adjacent.
    norm_pts = _normalize_pts(pts)
    if norm_pts is not None:
        return _points_to_params(norm_pts)
    return None

def _generate_halton_params(rng, seed_offset=0):
    """
    Generate parameters from a Halton sequence.
    Halton sequences are low-discrepancy, good for filling space.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((N, 2))
    for i in range(N):
        pts[i, 0] = halton(i + 1 + seed_offset, 2)
        pts[i, 1] = halton(i + 1 + seed_offset, 3)
    
    # Add slight random perturbation to break symmetry if needed, 
    # but Halton is deterministic.
    # Scale to roughly fit in [0,1]x[0,1]
    # Halton is already in [0,1]
    
    norm_pts = _normalize_pts(pts)
    if norm_pts is not None:
        return _points_to_params(norm_pts)
    return None

def _generate_random_params(rng):
    """Generate random parameters."""
    pts = rng.uniform(-1, 2, (N, 2))
    norm_pts = _normalize_pts(pts)
    if norm_pts is not None:
        return _points_to_params(norm_pts)
    return None

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    def _update_best(params):
        nonlocal best_params, best_score
        if params is None:
            return
        pts = _params_to_points(params)
        score = _compute_min_ratio(pts)
        if score > best_score:
            best_score = score
            best_params = params.copy()

    # Phase 1: Global Search using Differential Evolution
    # DE is robust to the non-smooth min() objective.
    # Bounds: The points should be in a reasonable range. 
    # Since we fix p0=(0,0) and p1=(1,0), the other points can be anywhere,
    # but for convex regions, they likely lie within a box like [-2, 2]x[-2, 2].
    bounds = [(-2.0, 2.0)] * (2 * (N - 2))
    
    # We run DE for a portion of the time.
    # maxiter and popsize need to be tuned for time.
    # Let's try to run DE for ~60 seconds, then local polish.
    
    de_start = time.time()
    de_time_budget = 60.0
    
    try:
        # Use a moderate population size for global exploration
        # popsize=15 is standard, but we can try 20 for better coverage.
        res_de = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            init='sobol', # Sobol initialization is better than 'latinhypercube' or 'halton' sometimes
            # We don't use 'workers' to keep it simple and avoid multiprocessing overhead
        )
        
        _update_best(res_de.x)
        
    except Exception as e:
        print(f"DE failed: {e}")
    
    elapsed = time.time() - start_time
    if elapsed > time_limit - 20.0:
        if best_params is not None:
            return _params_to_points(best_params)
        # Fallback
        reg_params = _generate_regular_polygon_params()
        if reg_params is not None:
            return _params_to_points(reg_params)
        return _params_to_points(rng.uniform(-1, 1, (2*(N-2),)))

    # Phase 2: Local Polish with Nelder-Mead
    # Nelder-Mead is derivative-free and handles kinks well.
    # We start from the best DE result and also try some structured starts.
    
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
    
    # Add structured starts
    reg_params = _generate_regular_polygon_params()
    if reg_params is not None:
        candidates.append(reg_params)
        
    # Add a few Halton starts
    for i in range(5):
        h_params = _generate_halton_params(rng, seed_offset=i*10)
        if h_params is not None:
            candidates.append(h_params)
            
    # Add a few random starts
    for i in range(5):
        r_params = _generate_random_params(rng)
        if r_params is not None:
            candidates.append(r_params)

    # Evaluate all candidates to find the best starting point for polishing
    # We already updated best_score in _update_best, but let's ensure we have a good list.
    # Actually, _update_best was called for DE result. Let's evaluate the others too.
    for p in candidates:
        _update_best(p)

    # Now polish the top candidates
    # Sort candidates by score? We just have best_params. 
    # Let's polish the best_params found so far.
    
    if best_params is not None:
        # Use Nelder-Mead for fine tuning
        # maxiter needs to be large enough but bounded by time.
        nm_start = time.time()
        nm_budget = time_limit - elapsed - 5.0 # Leave 5s for final check
        
        if nm_budget > 5.0:
            try:
                # Nelder-Mead options
                # xatol and fatol control convergence
                res_nm = minimize(
                    _objective,
                    best_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                # Check if it improved
                if res_nm.fun < _objective(best_params):
                    _update_best(res_nm.x)
                    
            except Exception as e:
                print(f"NM failed: {e}")
                
    # Phase 3: If we still have time, try a second round of DE from the best found?
    # Or just a tighter local search.
    # Let's try Powell method which is often better than NM for high dimensions.
    
    elapsed = time.time() - start_time
    if best_params is not None and elapsed < time_limit - 10.0:
        try:
            res_powell = minimize(
                _objective,
                best_params,
                method='Powell',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'xtol': 1e-10
                }
            )
            if res_powell.fun < _objective(best_params):
                _update_best(res_powell.x)
        except Exception as e:
            pass

    # Final Fallback
    if best_params is None:
        # Should not happen, but just in case
        best_params = _generate_regular_polygon_params()
        if best_params is None:
            best_params = np.zeros(2*(N-2))
            
    final_pts = _params_to_points(best_params)
    
    # One last check: ensure the points are valid (hull area > 0)
    if _compute_hull_area(final_pts) <= 0:
        # Return a known good configuration
        reg_params = _generate_regular_polygon_params()
        if reg_params is not None:
            final_pts = _params_to_points(reg_params)
            
    return final_pts

# EVOLVE-BLOCK-END
