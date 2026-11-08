"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

# Precompute all triple indices for vectorized area computation
_TRIPLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))
_NUM_TRIPLES = _TRIPLE_INDICES.shape[0]  # C(14,3) = 364

def _all_triangle_areas(points):
    """
    Compute all triangle areas vectorized.
    points: array of shape (N, 2)
    Returns: array of shape (NUM_TRIPLES,)
    """
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # 2D cross product: v1[0]*v2[1] - v1[1]*v2[0]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by three points.
    """
    areas = _all_triangle_areas(points)
    return np.min(areas)

def _convex_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """
    Score is min_triangle_area / convex_hull_area
    """
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_triangle_area(points, temperature=0.01):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    soft_min(x) = -1/T * log(sum(exp(-T*x)))
    """
    areas = _all_triangle_areas(points)
    # Avoid numerical issues
    areas = np.maximum(areas, 1e-12)
    neg_T_areas = -areas / temperature
    # Shift for numerical stability
    max_val = np.max(neg_T_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_T_areas - max_val)))
    soft_min = -temperature * log_sum_exp
    return soft_min

def _soft_score(points, temperature=0.01):
    """
    Smooth score using soft-min triangle area.
    """
    soft_min_area = _soft_min_triangle_area(points, temperature)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return soft_min_area / hull_area

def _negative_score(flat_points):
    """
    Objective function for minimizer (exact).
    """
    points = flat_points.reshape(N, 2)
    return -_score(points)

def _negative_soft_score(flat_points, temperature=0.01):
    """
    Smooth objective function for minimizer.
    """
    points = flat_points.reshape(N, 2)
    return -_soft_score(points, temperature)

def _generate_initializations():
    """Generate diverse initial point configurations."""
    candidates = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    candidates.append(np.column_stack((x, y)))
    
    # 2. Two concentric rings
    outer_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    x_outer = np.cos(outer_angles)
    y_outer = np.sin(outer_angles)
    x_inner = 0.5 * np.cos(inner_angles)
    y_inner = 0.5 * np.sin(inner_angles)
    candidates.append(np.column_stack((np.concatenate([x_outer, x_inner]), 
                                       np.concatenate([y_outer, y_inner]))))
    
    # 3. Grid (4x4, take 14)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    candidates.append(grid_points[:N].copy())
    
    # 4. Hexagonal packing pattern
    hex_points = []
    # Center
    hex_points.append([0.0, 0.0])
    # First ring (6 points)
    for ang in np.linspace(0, 2*np.pi, 6, endpoint=False):
        hex_points.append([0.4 * np.cos(ang), 0.4 * np.sin(ang)])
    # Second ring (7 more points to reach 14)
    for ang in np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7:
        hex_points.append([0.8 * np.cos(ang), 0.8 * np.sin(ang)])
    candidates.append(np.array(hex_points[:N]))
    
    # 5-8. Random with different seeds
    for seed in [42, 123, 456, 789]:
        rng = np.random.RandomState(seed)
        rand_points = rng.rand(N, 2) * 2 - 1  # Scale to [-1, 1]
        candidates.append(rand_points)
    
    return candidates

def _optimize_from_start(init_pts, deadline, use_soft=True):
    """
    Optimize from a single starting point.
    Uses soft-min surrogate first, then polishes with exact objective.
    """
    best_score = -np.inf
    best_points = init_pts.copy()
    
    # Phase 1: Soft-min guided optimization (if enabled)
    if use_soft:
        for temp in [0.1, 0.05, 0.02]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    lambda x: _negative_soft_score(x, temp),
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                if res.x is not None:
                    cur_pts = res.x.reshape(N, 2)
                    cur_score = _score(cur_pts)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_points = cur_pts
            except Exception:
                pass
    
    # Phase 2: Exact objective polish
    if time.time() < deadline:
        try:
            res = minimize(
                _negative_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.x is not None:
                cur_pts = res.x.reshape(N, 2)
                cur_score = _score(cur_pts)
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = cur_pts
        except Exception:
            pass
    
    # Phase 3: Final tight polish if time remains
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _negative_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            if res.x is not None:
                cur_pts = res.x.reshape(N, 2)
                cur_score = _score(cur_pts)
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = cur_pts
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    candidates = _generate_initializations()
    
    best_score = -np.inf
    best_points = None
    
    for init_pts in candidates:
        if time.time() > deadline - 10:  # Reserve 10s for final polish
            break
            
        try:
            pts, score = _optimize_from_start(init_pts, deadline)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        rng = np.random.RandomState(0)
        return rng.rand(N, 2)
        
    # Final global polish with remaining time
    if time.time() < deadline:
        try:
            res = minimize(
                _negative_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.x is not None:
                final_pts = res.x.reshape(N, 2)
                final_score = _score(final_pts)
                if final_score > best_score:
                    best_points = final_pts
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
