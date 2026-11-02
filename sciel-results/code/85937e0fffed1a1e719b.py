"""Heilbronn convex regions n=14 - centrally symmetric parameterization with multi-phase optimization."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
HALF_N = 7  # 7 pairs of antipodal points

def _min_triangle_area(points):
    """Compute minimum triangle area over all C(n,3) triples."""
    n = len(points)
    min_area = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1x = points[j, 0] - points[i, 0]
                v1y = points[j, 1] - points[i, 1]
                v2x = points[k, 0] - points[i, 0]
                v2y = points[k, 1] - points[i, 1]
                area = 0.5 * abs(v1x * v2y - v1y * v2x)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 0.0

def _score(points):
    """Compute min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _params_to_points(params):
    """Convert 14 parameters (7 angles + 7 radii) to 14 points.
    
    Parameters:
    - First 7: angles in (0, pi) for the 7 half-points
    - Last 7: radii for the 7 half-points
    
    The 14 points are: p_i = r_i * (cos(a_i), sin(a_i)) and -p_i for i=0..6
    """
    angles = params[:HALF_N]
    radii = params[HALF_N:]
    
    # Ensure angles are in valid range and sorted for convex position
    angles = np.clip(angles, 0.02, np.pi - 0.02)
    radii = np.clip(radii, 0.05, 3.0)
    
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    return all_points

def _objective(params):
    """Negative score for minimization."""
    points = _params_to_points(params)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    DEADLINE = 138.0  # Safety margin before 150s
    
    best_score = 0.0
    best_params = None
    
    # Bounds for differential evolution
    # Angles: (0.02, pi-0.02), Radii: (0.05, 3.0)
    bounds = [(0.02, np.pi - 0.02)] * HALF_N + [(0.05, 3.0)] * HALF_N
    
    # Phase 1: Differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=40,
            tol=1e-5,
            popsize=12,
            polish=False,
            init='sobol'
        )
        if result.fun is not None:
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 15:
        if best_params is not None:
            return _params_to_points(best_params)
        return np.array([[np.cos(2*np.pi*i/N), np.sin(2*np.pi*i/N)] for i in range(N)])
    
    # Phase 2: Local refinement from best DE result and some random starts
    start_points = []
    if best_params is not None:
        start_points.append(best_params)
    
    rng = np.random.RandomState(123)
    for _ in range(4):
        angles = np.sort(rng.uniform(0.1, np.pi - 0.1, HALF_N))
        radii = rng.uniform(0.2, 1.5, HALF_N)
        start_points.append(np.concatenate([angles, radii]))
    
    for sp in start_points:
        if time.time() - start_time > DEADLINE - 8:
            break
        try:
            res = minimize(_objective, sp, method='L-BFGS-B', bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            pass
    
    # Phase 3: Fine-tuning with Nelder-Mead
    if best_params is not None and time.time() - start_time < DEADLINE - 3:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            pass
    
    if best_params is not None:
        return _params_to_points(best_params)
    
    # Fallback: regular 14-gon
    return np.array([[np.cos(2*np.pi*i/N), np.sin(2*np.pi*i/N)] for i in range(N)])
