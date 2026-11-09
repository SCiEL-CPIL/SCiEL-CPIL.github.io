import numpy as np
from scipy.optimize import differential_evolution, minimize
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

def _params_to_points(params):
    """
    Convert parameters to points.
    params: array of shape (2*N,)
    [0:N] are angles (sorted), [N:2N] are radii.
    To ensure convexity and avoid self-intersection, we sort by angle.
    We normalize the scale later or let the optimizer handle it.
    For DE, we work in a normalized space.
    """
    angles = params[:N]
    radii = params[N:]
    
    # Sort by angle to maintain convex order
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    radii = radii[sort_idx]
    
    # Ensure angles are in [0, 2pi) and sorted
    # If angles are close, we might have issues, but DE explores broadly.
    # To prevent points from collapsing to the center, we enforce a minimum radius.
    radii = np.maximum(radii, 0.1)
    
    # Normalize angles to [0, 2pi]
    angles = np.mod(angles, 2 * np.pi)
    # Re-sort after mod to ensure continuity
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    radii = radii[sort_idx]
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    points = np.column_stack((x, y))
    
    # Center the points to remove translation invariance
    # This helps the optimizer focus on shape, not position
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Normalize scale to unit area? 
    # The score is ratio-based, so scale doesn't matter. 
    # However, for numerical stability in DE, we keep scale ~1.
    # Let's normalize so that max radius is 1.
    max_r = np.max(np.sqrt(x**2 + y**2))
    if max_r > 1e-6:
        points = points / max_r
        
    return points

def _negative_score_from_params(params):
    """
    Objective function for Differential Evolution.
    Returns negative score.
    """
    points = _params_to_points(params)
    score = _score(points)
    return -score

def _negative_score_cartesian(flat_points):
    """
    Objective function for L-BFGS-B polish.
    Directly uses Cartesian coordinates.
    """
    points = flat_points.reshape(N, 2)
    # Center points to remove translation invariance for gradient stability
    # Note: This is a transformation. If we center here, the gradient w.r.t flat_points
    # is slightly more complex but L-BFGS-B approximates it.
    # Actually, it's safer to just evaluate the score on the raw points.
    # The invariance might cause issues, but L-BFGS-B is robust enough if initialized well.
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    rng = np.random.default_rng(42)
    
    # 1. Global Search with Differential Evolution
    # Parameters: N angles + N radii
    # Angles in [0, 2pi], Radii in [0.1, 1.0]
    bounds = [(0, 2*np.pi)] * N + [(0.1, 1.0)] * N
    
    # Initial population: Regular polygon + noise
    # Regular polygon angles: 2*pi*i/N
    # Regular polygon radii: 1.0
    
    # DE requires a population. We let scipy generate it, but we can seed it.
    # To save time, we use a moderate population size and maxiter.
    
    best_params = None
    best_score_global = 0.0
    
    # Time check before starting DE
    if time.time() < deadline - 60:
        try:
            # Use 'differential_evolution'
            # tol=0.01 is loose for global search
            # maxiter=100 is reasonable
            # popsize=15 is small to save time, but DE is slow. 
            # Let's try popsize=10, maxiter=50.
            
            res_de = differential_evolution(
                _negative_score_from_params,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=0.05,
                polish=False,  # We will do our own polish
                updating='immediate'
            )
            
            if res_de.success or res_de.fun < -best_score_global:
                best_params = res_de.x
                best_score_global = -res_de.fun
                # Convert to points for tracking
                pts = _params_to_points(best_params)
                # Re-evaluate exact score
                s = _score(pts)
                if s > best_score_global:
                    best_score_global = s
        except Exception:
            pass

    # 2. Local Refinement with L-BFGS-B
    # We need a good starting point in Cartesian coordinates.
    
    # Start from the best DE result if available, else regular polygon
    if best_params is not None:
        start_points = _params_to_points(best_params)
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        start_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    # Center and normalize start_points
    centroid = np.mean(start_points, axis=0)
    start_points = start_points - centroid
    max_r = np.max(np.sqrt(start_points[:, 0]**2 + start_points[:, 1]**2))
    if max_r > 1e-6:
        start_points = start_points / max_r

    best_points = start_points.copy()
    best_score = _score(best_points)
    
    # Polish with L-BFGS-B
    # We use the exact score (non-smooth) but L-BFGS-B can handle it if the minimum is isolated.
    # Alternatively, we can use a very small temperature soft-min for the first polish step.
    
    # Strategy: 
    # 1. L-BFGS-B with exact score (min_area / hull_area)
    # This is non-differentiable at points where two triangle areas are equal min.
    # However, in practice, the global optimum often has a unique minimum triangle or a few degenerate ones.
    # L-BFGS-B might stall. 
    # Let's try a multi-step polish:
    # Step A: Soft-min with T=0.01 to smooth out
    # Step B: Soft-min with T=0.001
    # Step C: Exact score with tight tolerance
    
    if time.time() < deadline - 10:
        # Step A: Soft-min polish
        temp = 0.01
        def obj_soft(x):
            points = x.reshape(N, 2)
            areas = _all_triangle_areas(points)
            areas = np.maximum(areas, 1e-12)
            neg_T_areas = -areas / temp
            max_val = np.max(neg_T_areas)
            log_sum_exp = max_val + np.log(np.sum(np.exp(neg_T_areas - max_val)))
            soft_min_area = -temp * log_sum_exp
            
            hull_area = _convex_hull_area(points)
            if hull_area < 1e-10:
                return 1e6
            return -(soft_min_area / hull_area)
        
        try:
            res = minimize(
                obj_soft,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.fun < -best_score:
                new_points = res.x.reshape(N, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass

    if time.time() < deadline - 5:
        # Step B: Exact score polish
        # Use the exact score. 
        # To help L-BFGS-B with non-differentiability, we can use a small step size or just rely on the gradient approximation.
        
        def obj_exact(x):
            points = x.reshape(N, 2)
            return _negative_score_cartesian(x)
            
        try:
            res = minimize(
                obj_exact,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < -best_score:
                new_points = res.x.reshape(N, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass

    # Final check: Try a few random restarts of L-BFGS-B from the best point with small perturbations
    # This helps escape shallow local minima caused by non-differentiability.
    if time.time() < deadline - 2:
        for i in range(5):
            if time.time() > deadline - 2:
                break
            noise = rng.normal(0, 0.01, best_points.shape)
            perturbed = best_points + noise
            # Re-center
            centroid = np.mean(perturbed, axis=0)
            perturbed = perturbed - centroid
            max_r = np.max(np.sqrt(perturbed[:, 0]**2 + perturbed[:, 1]**2))
            if max_r > 1e-6:
                perturbed = perturbed / max_r
                
            try:
                res = minimize(
                    _negative_score_cartesian,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
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
