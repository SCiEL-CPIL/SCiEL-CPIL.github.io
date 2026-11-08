import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _triangle_areas_vectorized(points):
    """Compute areas of all C(N,3) triangles. Returns array of length C(N,3)."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    return np.array(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area_ratio(points):
    """Compute min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_ratio(points, temperature=0.01):
    """Smooth surrogate: soft-min of triangle areas / hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    # Soft minimum: -t * log(sum(exp(-area/t)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    soft_min = -temperature * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    return soft_min / hull_area

def construct():
    """
    Constructs 13 points to maximize the minimum triangle area divided by the convex hull area.
    Uses multi-phase optimization with diverse initial configurations.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_points = None
    best_score = -1.0

    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False

    def time_remaining():
        return deadline - time.time()

    # Generate diverse initial configurations
    def regular_polygon(n_points, radius=1.0, angle_offset=0.0):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False) + angle_offset
        return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

    def two_ring_configuration(inner_n, outer_n, inner_r=0.5, outer_r=1.0):
        """Points on two concentric circles."""
        inner_angles = np.linspace(0, 2 * np.pi, inner_n, endpoint=False)
        outer_angles = np.linspace(0, 2 * np.pi, outer_n, endpoint=False) + np.pi / outer_n
        inner_pts = np.column_stack((inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)))
        outer_pts = np.column_stack((outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)))
        return np.vstack([inner_pts, outer_pts])

    # Initial configurations to try
    initial_configs = []
    
    # Regular 13-gon
    initial_configs.append(regular_polygon(N))
    
    # Regular 13-gon with slight rotation
    initial_configs.append(regular_polygon(N, angle_offset=0.1))
    
    # Two-ring: 6 inner + 7 outer
    initial_configs.append(two_ring_configuration(6, 7, 0.5, 1.0))
    
    # Two-ring: 5 inner + 8 outer
    initial_configs.append(two_ring_configuration(5, 8, 0.6, 1.0))
    
    # Two-ring: 7 inner + 6 outer
    initial_configs.append(two_ring_configuration(7, 6, 0.4, 1.0))
    
    # Perturbed regular polygon
    rng = np.random.default_rng(42)
    for _ in range(3):
        noise = rng.normal(0, 0.05, (N, 2))
        initial_configs.append(regular_polygon(N) + noise)

    # Phase 1: Coarse optimization with soft-min surrogate
    for config_idx, init_pts in enumerate(initial_configs):
        if time_remaining() < 5.0:
            break
        
        # Evaluate initial
        score = _min_triangle_area_ratio(init_pts)
        update_best(init_pts, score)
        
        if time_remaining() < 5.0:
            break
        
        # Soft-min objective for gradient-based optimization
        def soft_min_objective(x):
            points = x.reshape((N, 2))
            return -_soft_min_ratio(points, temperature=0.02)
        
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                soft_min_objective, x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6},
            )
            
            if res.x is not None:
                pts = res.x.reshape((N, 2))
                score = _min_triangle_area_ratio(pts)
                update_best(pts, score)
        except Exception:
            pass

    # Phase 2: Fine optimization with exact objective
    # Start from best found so far and nearby configurations
    if best_points is not None:
        fine_starts = [best_points.copy()]
        
        # Add slightly perturbed versions
        rng2 = np.random.default_rng(123)
        for _ in range(4):
            noise = rng2.normal(0, 0.03, (N, 2))
            fine_starts.append(best_points + noise)
        
        # Also try from other good initial configs
        for config in initial_configs:
            score = _min_triangle_area_ratio(config)
            if score > best_score * 0.95:
                fine_starts.append(config.copy())
        
        for fine_idx, start_pts in enumerate(fine_starts):
            if time_remaining() < 10.0:
                break
            
            def exact_objective(x):
                points = x.reshape((N, 2))
                return -_min_triangle_area_ratio(points)
            
            x0 = start_pts.flatten()
            
            try:
                res = minimize(
                    exact_objective, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True},
                )
                
                if res.x is not None:
                    pts = res.x.reshape((N, 2))
                    score = _min_triangle_area_ratio(pts)
                    update_best(pts, score)
            except Exception:
                pass

    # Phase 3: Local refinement with small perturbations around best
    if best_points is not None and time_remaining() > 5.0:
        rng3 = np.random.default_rng(777)
        for _ in range(10):
            if time_remaining() < 3.0:
                break
            
            noise = rng3.normal(0, 0.01, (N, 2))
            perturbed = best_points + noise
            
            def local_objective(x):
                points = x.reshape((N, 2))
                return -_min_triangle_area_ratio(points)
            
            try:
                res = minimize(
                    local_objective, perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10},
                )
                
                if res.x is not None:
                    pts = res.x.reshape((N, 2))
                    score = _min_triangle_area_ratio(pts)
                    update_best(pts, score)
            except Exception:
                pass

    # Final safety: if we never found a valid point, return regular polygon
    if best_points is None:
        best_points = regular_polygon(N)

    return best_points
