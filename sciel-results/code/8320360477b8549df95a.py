import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _compute_areas(points):
    """Compute all 364 triangle areas efficiently using vectorized cross products."""
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    return 0.5 * np.abs(cross)

def _calculate_score(points):
    """Calculate the score: min triangle area / convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    areas = _compute_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min_score(points, temperature=0.01):
    """Soft-min objective for smoother optimization landscape."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    areas = _compute_areas(points)
    # Soft minimum: approximates min while being differentiable
    soft_min = -temperature * np.log(np.sum(np.exp(-areas / temperature)))
    
    return soft_min / hull_area

def _two_temp_soft_min(points, temp1=0.005, temp2=0.02):
    """Two-temperature soft-min for better gradient information."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    areas = _compute_areas(points)
    sm1 = -temp1 * np.log(np.sum(np.exp(-areas / temp1)))
    sm2 = -temp2 * np.log(np.sum(np.exp(-areas / temp2)))
    
    # Blend: use softer min for exploration, sharper for exploitation
    return (0.7 * sm1 + 0.3 * sm2) / hull_area

def _generate_two_ring(r_inner, theta_offset, r_outer=1.0):
    """Generate two-ring configuration: 7 outer + 7 inner points with 7-fold symmetry."""
    n_out, n_in = 7, 7
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta_offset
    
    outer = np.column_stack((r_outer * np.cos(angles_out), r_outer * np.sin(angles_out)))
    inner = np.column_stack((r_inner * np.cos(angles_in), r_inner * np.sin(angles_in)))
    
    return np.vstack([outer, inner])

def _score_ring_params(params):
    """Score function for two-ring parameters: [r_inner, theta_offset]."""
    r_inner, theta_offset = params
    points = _generate_two_ring(r_inner, theta_offset)
    return _calculate_score(points)

def _soft_score_ring_params(params, temp=0.01):
    """Soft-min score for ring params."""
    r_inner, theta_offset = params
    points = _generate_two_ring(r_inner, theta_offset)
    return _soft_min_score(points, temp)

def _optimize_two_ring(deadline):
    """Optimize two-ring parameters using differential evolution + local polish."""
    best_score = 0.0
    best_params = None
    
    # Phase 1: Global search with differential evolution
    try:
        res = differential_evolution(
            lambda p: -_score_ring_params(p),
            bounds=[(0.2, 0.95), (0.0, 2*np.pi)],
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-8,
            polish=False
        )
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Phase 2: Local polish with Nelder-Mead
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_nm = minimize(
                lambda p: -_score_ring_params(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
    
    if best_params is None:
        return 0.0, None
    
    return best_score, best_params

def _coordinate_polish(points, deadline):
    """Coordinate-descent polish to escape parameterization local optima."""
    best_score = _calculate_score(points)
    best_points = points.copy()
    step_size = 0.05
    
    while time.time() < deadline - 2:
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try moving in positive and negative directions
                for direction in [1, -1]:
                    candidate = best_points.copy()
                    candidate[i, coord] += direction * step_size
                    
                    # Ensure point stays in reasonable bounds
                    if np.linalg.norm(candidate[i]) > 2.0:
                        continue
                    
                    cand_score = _calculate_score(candidate)
                    if cand_score > best_score + 1e-12:
                        best_score = cand_score
                        best_points = candidate
                        improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_score, best_points

def _polish_points(points, deadline):
    """Full polish: Nelder-Mead on all coordinates + coordinate descent."""
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    # Phase 1: Nelder-Mead on all 28 coordinates
    if time.time() < deadline - 5:
        try:
            def neg_score(x):
                pts = x.reshape(-1, 2)
                return -_calculate_score(pts)
            
            res = minimize(
                neg_score,
                points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1500, 'adaptive': True}
            )
            
            candidate_score = -res.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = res.x.reshape(-1, 2)
        except Exception:
            pass
    
    # Phase 2: Coordinate descent polish
    if time.time() < deadline - 3:
        cd_score, cd_points = _coordinate_polish(best_points, deadline)
        if cd_score > best_score:
            best_score = cd_score
            best_points = cd_points
    
    return best_score, best_points

def _optimize_single_ring(deadline):
    """Optimize single 14-gon with variable radii (fallback strategy)."""
    def score_params(params):
        # params: 14 radii
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack((params * np.cos(angles), params * np.sin(angles)))
        return _calculate_score(pts)
    
    best_score = 0.0
    best_params = None
    
    try:
        res = differential_evolution(
            lambda p: -score_params(p),
            bounds=[(0.5, 1.5)] * N,
            seed=789,
            maxiter=30,
            popsize=15,
            tol=1e-6,
            polish=False
        )
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_nm = minimize(
                lambda p: -score_params(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
    
    if best_params is None:
        return 0.0, None
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack((best_params * np.cos(angles), best_params * np.sin(angles)))
    return best_score, points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Strategy 1: Two-ring (7+7) - primary strategy
    if time.time() < deadline - 30:
        ring_score, ring_params = _optimize_two_ring(deadline - 20)
        if ring_score > best_overall_score and ring_params is not None:
            best_overall_score = ring_score
            best_overall_points = _generate_two_ring(*ring_params)
    
    # Strategy 2: Single ring with variable radii
    if time.time() < deadline - 25 and best_overall_points is not None:
        ring2_score, ring2_points = _optimize_single_ring(deadline - 15)
        if ring2_score > best_overall_score:
            best_overall_score = ring2_score
            best_overall_points = ring2_points
    
    # Fallback: regular 14-gon
    if best_overall_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_overall_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_overall_score = _calculate_score(best_overall_points)
    
    # Final polish on all 28 coordinates
    if time.time() < deadline - 5:
        polish_score, polish_points = _polish_points(best_overall_points, deadline)
        if polish_score > best_overall_score:
            best_overall_score = polish_score
            best_overall_points = polish_points
    
    return best_overall_points
