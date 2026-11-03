"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

# Time budget
TIME_LIMIT = 130.0  # seconds (safety margin before 150s kill)
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return TIME_LIMIT - _elapsed()

def _min_triangle_area_normalized(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the ratio (higher is better).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-15:
        return 0.0
    
    # Find minimum triangle area over all triples
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    
    if min_area <= 1e-15:
        return 0.0
    
    return min_area / hull_area

def _neg_score(points):
    """Negative of the score for minimization."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return -_min_triangle_area_normalized(pts)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    def _update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Strategy 1: Start with regular 14-gon and optimize
    # Regular 14-gon vertices
    angles = 2 * np.pi * np.arange(N) / N
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    try:
        hull = ConvexHull(regular)
        area = hull.volume
        if area > 0:
            regular = regular / np.sqrt(area)
    except Exception:
        pass
    
    score_regular = _min_triangle_area_normalized(regular)
    _update_best(regular, score_regular)
    
    # Strategy 2: Differential evolution on the 14 points (28 dimensions)
    if _remaining() > 20:
        # Bounds: points in [-2, 2]^2 (will be normalized)
        bounds = [(-2.0, 2.0)] * (2 * N)
        
        def _de_objective(x):
            pts = x.reshape(N, 2)
            # Normalize to unit area for fair comparison
            try:
                hull = ConvexHull(pts)
                area = hull.volume
                if area <= 1e-15:
                    return 1.0  # Bad
                pts_norm = pts / np.sqrt(area)
            except Exception:
                return 1.0
            return -_min_triangle_area_normalized(pts_norm)
        
        try:
            result = differential_evolution(
                _de_objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol',
                update_interval=1,
                callback=lambda xk, convergence: (
                    None if _remaining() > 5 else (_de_objective(xk) if False else None)
                )
            )
            if result.success or result.fun < 0.5:
                pts_de = result.x.reshape(N, 2)
                try:
                    hull = ConvexHull(pts_de)
                    area = hull.volume
                    if area > 1e-15:
                        pts_de = pts_de / np.sqrt(area)
                except Exception:
                    pass
                score_de = _min_triangle_area_normalized(pts_de)
                _update_best(pts_de, score_de)
        except Exception:
            pass
    
    # Strategy 3: Local refinement using L-BFGS-B
    if _remaining() > 15 and best_points is not None:
        # Try multiple starting points
        start_points = [best_points]
        
        # Add some perturbed versions
        rng = np.random.RandomState(42)
        for _ in range(3):
            perturbed = best_points + rng.randn(N, 2) * 0.1
            start_points.append(perturbed)
        
        for start in start_points:
            if _remaining() < 10:
                break
            
            def _local_obj(x):
                pts = x.reshape(N, 2)
                # Normalize
                try:
                    hull = ConvexHull(pts)
                    area = hull.volume
                    if area <= 1e-15:
                        return 1.0
                    pts_norm = pts / np.sqrt(area)
                except Exception:
                    return 1.0
                return -_min_triangle_area_normalized(pts_norm)
            
            try:
                res = minimize(
                    _local_obj,
                    start.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-3.0, 3.0)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < 0.5:
                    pts_local = res.x.reshape(N, 2)
                    try:
                        hull = ConvexHull(pts_local)
                        area = hull.volume
                        if area > 1e-15:
                            pts_local = pts_local / np.sqrt(area)
                    except Exception:
                        pass
                    score_local = _min_triangle_area_normalized(pts_local)
                    _update_best(pts_local, score_local)
            except Exception:
                pass
    
    # Strategy 4: Support function approach with centrally symmetric structure
    # 7 pairs of antipodal points, parameterized by 7 radii and 7 angles
    if _remaining() > 10:
        # Use 7 independent angles (not necessarily uniform) and 7 radii
        # Points: for i in 0..6, point_i = r_i * (cos(a_i), sin(a_i)), point_{i+7} = -point_i
        
        def _symmetric_points(params):
            # params: 14 values: 7 angles, 7 radii
            angles = params[:7]
            radii = params[7:]
            angles = np.sort(angles)  # Ensure ordering
            # Make sure angles are in [0, pi) for unique representation
            angles = angles % np.pi
            angles = np.sort(angles)
            
            pts = np.zeros((N, 2))
            for i in range(7):
                x = radii[i] * np.cos(angles[i])
                y = radii[i] * np.sin(angles[i])
                pts[i] = [x, y]
                pts[i + 7] = [-x, -y]
            return pts
        
        def _symmetric_obj(params):
            pts = _symmetric_points(params)
            try:
                hull = ConvexHull(pts)
                area = hull.volume
                if area <= 1e-15:
                    return 1.0
                pts_norm = pts / np.sqrt(area)
            except Exception:
                return 1.0
            return -_min_triangle_area_normalized(pts_norm)
        
        # Initial guess: regular 7-gon in first half
        init_angles = np.pi * np.arange(7) / 7
        init_radii = np.ones(7)
        init_params = np.concatenate([init_angles, init_radii])
        
        bounds_sym = [(0, np.pi)] * 7 + [(0.1, 2.0)] * 7
        
        try:
            res_sym = minimize(
                _symmetric_obj,
                init_params,
                method='L-BFGS-B',
                bounds=bounds_sym,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_sym.fun < 0.5:
                pts_sym = _symmetric_points(res_sym.x)
                try:
                    hull = ConvexHull(pts_sym)
                    area = hull.volume
                    if area > 1e-15:
                        pts_sym = pts_sym / np.sqrt(area)
                except Exception:
                    pass
                score_sym = _min_triangle_area_normalized(pts_sym)
                _update_best(pts_sym, score_sym)
        except Exception:
            pass
    
    # Strategy 5: More aggressive differential evolution if time permits
    if _remaining() > 30:
        bounds = [(-2.0, 2.0)] * (2 * N)
        
        def _de_objective2(x):
            pts = x.reshape(N, 2)
            try:
                hull = ConvexHull(pts)
                area = hull.volume
                if area <= 1e-15:
                    return 1.0
                pts_norm = pts / np.sqrt(area)
            except Exception:
                return 1.0
            return -_min_triangle_area_normalized(pts_norm)
        
        try:
            result2 = differential_evolution(
                _de_objective2,
                bounds=bounds,
                seed=123,
                maxiter=100,
                popsize=20,
                tol=1e-7,
                polish=True,
                init='latinhypercube'
            )
            if result2.fun < 0.5:
                pts_de2 = result2.x.reshape(N, 2)
                try:
                    hull = ConvexHull(pts_de2)
                    area = hull.volume
                    if area > 1e-15:
                        pts_de2 = pts_de2 / np.sqrt(area)
                except Exception:
                    pass
                score_de2 = _min_triangle_area_normalized(pts_de2)
                _update_best(pts_de2, score_de2)
        except Exception:
            pass
    
    # Final: ensure we return a valid point set
    if best_points is None:
        # Fallback: regular 14-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        try:
            hull = ConvexHull(best_points)
            area = hull.volume
            if area > 0:
                best_points = best_points / np.sqrt(area)
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
