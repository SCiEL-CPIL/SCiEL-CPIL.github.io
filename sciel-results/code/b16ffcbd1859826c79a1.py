"""Differential Evolution + L-BFGS-B polish for Heilbronn convex regions n=14."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

_TRIPLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))
_NUM_TRIPLES = _TRIPLE_INDICES.shape[0]  # 364

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

def _soft_score(points, temperature=0.01):
    soft_min_area = _soft_min_triangle_area(points, temperature)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return soft_min_area / hull_area

def _params_to_points(params):
    """
    Convert polar perturbation parameters to point coordinates.
    params: shape (2*N,) = [delta_r_0, delta_theta_0, ..., delta_r_13, delta_theta_13]
    Base: regular 14-gon, radius 1.
    Constraints:
      - delta_r in [-0.4, 0.4]  => radius in [0.6, 1.4] (positive)
      - delta_theta in [-0.25, 0.25] => angular gaps remain positive (convexity)
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_radius = np.ones(N)
    
    delta_r = params[0::2]
    delta_theta = params[1::2]
    
    radii = base_radius + delta_r
    angles = base_angles + delta_theta
    
    # Ensure angles are in [0, 2*pi) and sorted for proper convex ordering
    # For small perturbations, order is preserved, but let's be safe
    # Actually, for convexity, we just need the points in CCW order with positive cross products
    # The parametrization with small angular perturbations around a regular polygon
    # guarantees this as long as angular gaps > 0.
    # Base gap = 2*pi/14 ~ 0.4488. Max delta_theta = 0.25.
    # Worst case gap: 0.4488 - 0.25 - 0.25 = -0.0512. This can be negative!
    # So we need to be careful. Let's use a smaller bound or enforce sorting.
    # If we sort by angle, we might break the "regular polygon perturbation" structure
    # but it ensures valid convex hull if radii are positive.
    
    # Let's clip delta_theta to [-0.2, 0.2] to be safer.
    # Min gap: 0.4488 - 0.2 - 0.2 = 0.0488 > 0. Good.
    
    # Re-clip for safety in case of numerical issues
    delta_r = np.clip(delta_r, -0.4, 0.4)
    delta_theta = np.clip(delta_theta, -0.2, 0.2)
    
    radii = base_radius + delta_r
    angles = base_angles + delta_theta
    
    # Normalize angles to [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    
    # Sort by angle to ensure CCW ordering
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    radii_sorted = radii[sort_idx]
    
    x = radii_sorted * np.cos(angles_sorted)
    y = radii_sorted * np.sin(angles_sorted)
    
    return np.column_stack((x, y))

def _negative_soft_score_params(params, temperature=0.01):
    """Objective for parameter-space optimization (soft)."""
    points = _params_to_points(params)
    return -_soft_score(points, temperature)

def _negative_score_params(params):
    """Objective for parameter-space optimization (exact)."""
    points = _params_to_points(params)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (killed at 150s)
    
    # Define bounds for differential evolution
    # delta_r: [-0.4, 0.4], delta_theta: [-0.2, 0.2]
    bounds = []
    for i in range(N):
        bounds.append((-0.4, 0.4))  # delta_r
        bounds.append((-0.2, 0.2))  # delta_theta
    
    best_score = -np.inf
    best_points = None
    
    # Phase 1: Differential Evolution (Global Search)
    # DE is slow, so we limit maxiter and popsize
    # We want to spend about 80% of the time on DE, 20% on polish
    de_deadline = start_time + 90.0
    
    try:
        # Use a custom callback to check time
        def de_callback(xk, convergence=True):
            return time.time() > de_deadline
        
        # DE settings:
        # popsize: 10-20 is reasonable for 28 dims
        # maxiter: limited by time
        # tol: not critical, we stop by time
        # seed: fixed for reproducibility
        de_result = differential_evolution(
            _negative_soft_score_params,
            bounds,
            args=(0.05,),  # temperature for soft min
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            callback=de_callback,
            polish=False,  # We'll do our own polish
            init='latinhypercube'
        )
        
        if de_result.success or de_result.x is not None:
            best_params_de = de_result.x
            best_points_de = _params_to_points(best_params_de)
            score_de = _score(best_points_de)
            if score_de > best_score:
                best_score = score_de
                best_points = best_points_de
    except Exception:
        pass
    
    # If DE didn't find anything good, fallback to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)
    
    # Phase 2: Local Polish with L-BFGS-B on soft-min objective
    # Start from the best DE result
    if time.time() < deadline - 5:
        try:
            # Convert best_points back to params? No, we have best_params_de
            # But we might have updated best_points from fallback.
            # Let's just use Cartesian space for polish to be safe, or re-derive params.
            # Re-deriving params from points is tricky. Let's polish in Cartesian space.
            
            # Actually, polishing in parameter space is better constrained.
            # If we used DE, we have best_params_de.
            # If we used fallback, we can create zero params.
            
            # Let's try to polish the DE result in parameter space
            if 'best_params_de' in locals() and de_result.x is not None:
                init_params = de_result.x
            else:
                init_params = np.zeros(2 * N)
            
            # L-BFGS-B on soft score with low temperature
            temp_polish = 0.01
            res = minimize(
                lambda x: _negative_soft_score_params(x, temp_polish),
                init_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.x is not None:
                cur_points = _params_to_points(res.x)
                cur_score = _score(cur_points)
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = cur_points
        except Exception:
            pass
    
    # Phase 3: Final Exact Polish in Cartesian Space (if time remains)
    # This allows escaping the parameterization constraints slightly if needed
    if time.time() < deadline - 1:
        try:
            # Use Nelder-Mead for the non-smooth exact objective in Cartesian space
            # But Nelder-Mead is slow. Let's use a few steps of L-BFGS-B on a very soft min
            # or just stick with the parameter space result.
            
            # Actually, let's do a short Nelder-Mead in Cartesian space for final tweak
            # but limit iterations strictly
            remaining_time = deadline - time.time()
            if remaining_time > 2:
                # Estimate max iterations based on time
                # Each eval is fast, but NM is iterative
                max_iter_nm = int(remaining_time * 100)  # Rough estimate
                max_iter_nm = max(10, min(max_iter_nm, 2000))
                
                res = minimize(
                    lambda flat_pts: -_score(flat_pts.reshape(N, 2)),
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_nm, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                if res.x is not None:
                    final_pts = res.x.reshape(N, 2)
                    final_score = _score(final_pts)
                    if final_score > best_score:
                        best_score = final_score
                        best_points = final_pts
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
