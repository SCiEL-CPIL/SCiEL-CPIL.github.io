"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices for vectorized area calculation
_TRIPLETS = np.array(list(combinations(range(N), 3)))

def _triangle_areas(pts):
    """
    Compute all triangle areas efficiently using precomputed indices.
    pts: (N, 2) array
    """
    i = _TRIPLETS[:, 0]
    j = _TRIPLETS[:, 1]
    k = _TRIPLETS[:, 2]
    v1 = pts[j] - pts[i]
    v2 = pts[k] - pts[i]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _smooth_min(values, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_val = np.min(values)
    shifted = -tau * (values - min_val)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50, 0)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_val - (1.0 / tau) * log_sum

def _params_to_pts(params):
    """
    Convert 22 free parameters to full (13, 2) point array.
    p0 is fixed at (0,0), p1 is fixed at (1,0).
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    return pts

def _objective(params, tau):
    """
    Objective function to minimize: -smooth_min(triangle_areas / hull_area).
    """
    pts = _params_to_pts(params)
    hull_area = _convex_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    areas = _triangle_areas(pts)
    ratios = areas / hull_area
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _pts_to_params(pts):
    """
    Transform points to fixed frame: p0=(0,0), p1=(1,0).
    Returns 22 parameters for p2...p12.
    """
    # Translate
    pts = pts - pts[0]
    # Scale and Rotate so p1 becomes (1, 0)
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return None
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    # Ensure numerical stability for p0 and p1
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    return pts[2:].flatten()

def _generate_initial_params(rng, method='polygon'):
    """
    Generate initial parameters in the fixed frame.
    """
    if method == 'polygon':
        # Regular 13-gon, centered at origin
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # Add slight perturbation to break symmetry
        angles += rng.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng.normal(0, 0.05, N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        params = _pts_to_params(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)
        
    elif method == 'halton':
        # Halton sequence
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
            pts[i, 0] = halton(i + 1, 2)
            pts[i, 1] = halton(i + 1, 3)
        pts += rng.normal(0, 0.05, (N, 2))
        params = _pts_to_params(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)

    elif method == 'random':
        # Random points in a square, then transform
        pts = rng.uniform(-1, 1, (N, 2))
        # Ensure p0 and p1 are distinct enough
        if np.linalg.norm(pts[1] - pts[0]) < 0.1:
            pts[1] = pts[0] + rng.uniform(0.5, 1.0, 2)
        params = _pts_to_params(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)

    return rng.uniform(-1, 1, 22)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 0.0
    best_params = None
    best_pts = None
    
    # Define bounds for differential evolution
    # In the fixed frame, points shouldn't be too far from origin relative to p1-p0 distance
    # If they are too far, hull area grows, but min area might not grow proportionally.
    # A reasonable bound is [-2, 2]
    bounds = [(-2.0, 2.0)] * 22
    
    # Phase 1: Global Optimization using Differential Evolution
    # DE is robust to local minima but slower. We limit iterations and population.
    try:
        if time.time() < deadline - 40:
            # Use a moderate population size and max iterations
            # 22 dimensions, popsize 10 -> 220 individuals
            de_result = differential_evolution(
                _objective, 
                bounds, 
                args=(50.0,),  # Initial tau
                maxiter=50, 
                popsize=10,
                tol=1e-6,
                seed=42,
                polish=False  # We will do our own polish
            )
            if de_result.success or de_result.fun < 1e8:
                candidate_params = de_result.x
                candidate_pts = _params_to_pts(candidate_params)
                candidate_score = _score(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate_params.copy()
                    best_pts = candidate_pts.copy()
    except Exception:
        pass

    # Phase 2: Local Polish from multiple starts
    # We will use L-BFGS-B with increasing tau (sharpness)
    
    # Start candidates:
    # 1. Best from DE
    # 2. Regular Polygon
    # 3. Halton
    # 4. Random
    
    start_candidates = []
    if best_params is not None:
        start_candidates.append(best_params)
    
    rng = np.random.RandomState(0)
    start_candidates.append(_generate_initial_params(rng, 'polygon'))
    start_candidates.append(_generate_initial_params(rng, 'halton'))
    start_candidates.append(_generate_initial_params(rng, 'random'))
    
    # Add a few more random starts if time permits
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        start_candidates.append(_generate_initial_params(rng, 'random'))

    for params0 in start_candidates:
        if time.time() > deadline - 10:
            break
            
        current_params = params0.copy()
        
        # Progressive refinement with increasing tau
        # Start with soft min, end with hard min
        taus = [20.0, 50.0, 100.0, 200.0]
        
        for tau in taus:
            if time.time() > deadline - 5:
                break
                
            try:
                res = minimize(
                    _objective,
                    current_params,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < 1e8:
                    candidate_params = res.x
                    candidate_pts = _params_to_pts(candidate_params)
                    candidate_score = _score(candidate_pts)
                    
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_params = candidate_params.copy()
                        best_pts = candidate_pts.copy()
                        
                    current_params = candidate_params
            except Exception:
                break

    # Final check
    if best_params is not None:
        # One final high-precision polish on the absolute best
        if time.time() < deadline - 2:
            try:
                res = minimize(
                    _objective,
                    best_params,
                    args=(500.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                candidate_params = res.x
                candidate_pts = _params_to_pts(candidate_params)
                candidate_score = _score(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
            except Exception:
                pass

    if best_pts is None:
        # Fallback: Regular polygon
        best_pts = _params_to_pts(_generate_initial_params(np.random.RandomState(0), 'polygon'))

    return best_pts

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# We wrap it in the required structure if necessary, but the task says 
# "Keep the function names and signatures of the seed program".
# The seed program had construct() inside EVOLVE-BLOCK.
# I will place the entire logic inside the block as requested by the format, 
# but ensure the imports are at the top level if not allowed inside.
# However, the prompt shows imports INSIDE the block in the current program.
# So I will provide the full script.
# EVOLVE-BLOCK-END
