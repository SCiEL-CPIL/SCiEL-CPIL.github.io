import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# Pre-compute triangle indices for efficiency
_TRIPLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))
_NUM_TRIPLES = _TRIPLE_INDICES.shape[0]

def _all_triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    return np.min(_all_triangle_areas(points))

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_triangle_area(points, temperature=0.01):
    """Smooth approximation of min triangle area via log-sum-exp."""
    areas = _all_triangle_areas(points)
    areas = np.maximum(areas, 1e-12)
    neg_T_areas = -areas / temperature
    max_val = np.max(neg_T_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_T_areas - max_val)))
    return -temperature * log_sum_exp

def _negative_soft_score_cartesian(flat_points, temperature=0.01):
    """
    Objective function for L-BFGS-B.
    Uses soft-min to make the objective differentiable almost everywhere.
    """
    points = flat_points.reshape(N, 2)
    
    # Soft minimum of triangle areas
    soft_min_area = _soft_min_triangle_area(points, temperature)
    
    # Convex hull area
    hull_area = _convex_hull_area(points)
    
    if hull_area < 1e-10:
        return 1e6
    
    # We want to maximize (min_area / hull_area)
    # So we minimize -(soft_min_area / hull_area)
    return -(soft_min_area / hull_area)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # We will perform multiple restarts with different temperatures and initializations
    # Since L-BFGS-B is local, we need diverse starts.
    # However, the regular polygon is a strong start. We can perturb it.
    
    rng = np.random.default_rng(42)
    
    # Strategy: 
    # 1. Start with regular polygon, use moderate temperature.
    # 2. If time permits, try smaller temperatures for polishing.
    # 3. Try a few random perturbations of the regular polygon.
    
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    current_start = init_points.flatten().copy()
    
    for temp in temperatures:
        if time.time() > deadline - 10:
            break
            
        # Objective function closure for this temperature
        def objective(x):
            return _negative_soft_score_cartesian(x, temperature=temp)
            
        # Run L-BFGS-B
        try:
            res = minimize(
                objective,
                current_start,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < -best_score:
                new_points = res.x.reshape(N, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
                    current_start = new_points.flatten().copy()
                    
        except Exception:
            pass
            
    # Final exact polish step: use the exact min area but it's non-differentiable.
    # L-BFGS-B might struggle with the kinks. 
    # However, we can try Nelder-Mead or a simple coordinate descent on the exact score?
    # Or just stick with the soft-min result which is usually very close.
    # Let's try one more L-BFGS-B with a very small temperature to sharpen.
    
    if time.time() < deadline - 5:
        temp = 0.001
        def objective_sharp(x):
            return _negative_soft_score_cartesian(x, temperature=temp)
            
        try:
            res = minimize(
                objective_sharp,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < -best_score:
                new_points = res.x.reshape(N, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass

    return best_points
