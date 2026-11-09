import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

def compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def get_score(points):
    """Compute score: min triangle area / convex hull area."""
    N = len(points)
    if N < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0
    
    min_area = compute_min_triangle_area(points)
    if min_area <= 0:
        return 0.0
    return min_area / hull_area

def soft_min(areas, tau):
    """
    Compute the soft minimum of an array of areas using log-sum-exp trick.
    soft_min(x) = min_val - 1/tau * log(sum(exp(-tau * (x - min_val))))
    """
    min_val = np.min(areas)
    shifted = areas - min_val
    # Clip to avoid overflow in exp
    shifted = np.maximum(shifted, -50.0 / tau)
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    return min_val - (1.0 / tau) * log_sum

def compute_all_areas(points):
    """Compute all triangle areas for n points."""
    n = len(points)
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    N = 13
    best_score = 0.0
    best_points = None
    
    def time_left():
        return deadline - time.time()
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        s = get_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Strategy 1: Optimize angles of points on the unit circle (convex polygon)
    # This is the baseline approach that works well
    
    def angles_to_points(angles):
        """Convert angles to points on unit circle."""
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    def params_to_points_circle(params):
        """
        Convert params (angle differences) to points on unit circle.
        params shape: (N-2,)
        """
        d = np.maximum(params, 1e-6)
        sum_d = np.sum(d)
        if sum_d >= 2 * np.pi:
            scale = (2 * np.pi - 1e-6) / sum_d
            d = d * scale
        
        angles = np.zeros(N)
        angles[0] = 0.0
        angles[1:] = np.cumsum(d)
        
        return angles_to_points(angles)
    
    def points_to_params_circle(pts):
        """Convert points to angle differences."""
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        angles = angles % (2 * np.pi)
        angles = np.sort(angles)
        
        diffs = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
        return diffs[:N-2]
    
    def objective_circle(params, tau):
        """Minimize negative soft-min score for circle-based points."""
        try:
            pts = params_to_points_circle(params)
            areas = compute_all_areas(pts)
            sm = soft_min(areas, tau)
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
            except:
                return 1e6
            score = sm / hull_area
            return -score
        except:
            return 1e6
    
    # Initial guess: Regular N-gon
    initial_params_circle = np.full(N-2, 2 * np.pi / N)
    
    # Run progressive tau optimization on circle
    current_params_circle = initial_params_circle.copy()
    for tau in [20, 50, 100, 200]:
        if time_left() < 10.0:
            break
        try:
            res = minimize(lambda p: objective_circle(p, tau), current_params_circle, 
                         method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < 1e5:
                current_params_circle = res.x
                pts = params_to_points_circle(current_params_circle)
                update_best(pts)
        except:
            pass
    
    # Strategy 2: Full 2D optimization with multiple topologies
    # Topology A: 13-gon (all points on convex hull)
    # Topology B: 12-gon + 1 center point
    # Topology C: 11-gon + 2 interior points
    
    def make_regular_ngon(n):
        """Create regular n-gon centered at origin."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    def objective_2d(pts_flat, tau, topology):
        """
        Minimize negative soft-min score for 2D points.
        pts_flat: 2D array of shape (N, 2)
        topology: 'convex', 'center', or 'two_interior'
        """
        try:
            pts = pts_flat.reshape(N, 2)
            areas = compute_all_areas(pts)
            sm = soft_min(areas, tau)
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
            except:
                return 1e6
            score = sm / hull_area
            return -score
        except:
            return 1e6
    
    # Topology A: 13-gon (start from circle result)
    if best_points is not None and time_left() > 30.0:
        pts_init = best_points.copy()
        # Normalize to unit area
        try:
            hull = ConvexHull(pts_init)
            hull_area = hull.volume
            if hull_area > 1e-10:
                pts_init = pts_init / np.sqrt(hull_area)
        except:
            pts_init = make_regular_ngon(13)
        
        current_pts_A = pts_init.flatten().copy()
        for tau in [50, 100, 200]:
            if time_left() < 10.0:
                break
            try:
                res = minimize(lambda p: objective_2d(p, tau, 'convex'), current_pts_A,
                             method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
                if res.fun < 1e5:
                    current_pts_A = res.x
                    pts = current_pts_A.reshape(N, 2)
                    update_best(pts)
            except:
                pass
    
    # Topology B: 12-gon + center
    if time_left() > 25.0:
        # Initialize: 12-gon + center at origin
        pts_B = np.vstack([make_regular_ngon(12), [[0.0, 0.0]]])
        # Normalize
        try:
            hull = ConvexHull(pts_B)
            hull_area = hull.volume
            if hull_area > 1e-10:
                pts_B = pts_B / np.sqrt(hull_area)
        except:
            pass
        
        current_pts_B = pts_B.flatten().copy()
        for tau in [50, 100]:
            if time_left() < 10.0:
                break
            try:
                res = minimize(lambda p: objective_2d(p, tau, 'center'), current_pts_B,
                             method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-12})
                if res.fun < 1e5:
                    current_pts_B = res.x
                    pts = current_pts_B.reshape(N, 2)
                    update_best(pts)
            except:
                pass
    
    # Topology C: 11-gon + 2 interior points
    if time_left() > 20.0:
        # Initialize: 11-gon + 2 points near center
        pts_C = np.vstack([make_regular_ngon(11), [[0.2, 0.2], [-0.2, -0.2]]])
        try:
            hull = ConvexHull(pts_C)
            hull_area = hull.volume
            if hull_area > 1e-10:
                pts_C = pts_C / np.sqrt(hull_area)
        except:
            pass
        
        current_pts_C = pts_C.flatten().copy()
        for tau in [50]:
            if time_left() < 10.0:
                break
            try:
                res = minimize(lambda p: objective_2d(p, tau, 'two_interior'), current_pts_C,
                             method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                if res.fun < 1e5:
                    current_pts_C = res.x
                    pts = current_pts_C.reshape(N, 2)
                    update_best(pts)
            except:
                pass
    
    # Strategy 3: Differential evolution for global search
    if time_left() > 40.0:
        def de_objective(params_flat):
            try:
                pts = params_flat.reshape(N, 2)
                # Normalize to unit area
                try:
                    hull = ConvexHull(pts)
                    hull_area = hull.volume
                    if hull_area < 1e-10:
                        return 1e6
                    pts = pts / np.sqrt(hull_area)
                except:
                    return 1e6
                
                areas = compute_all_areas(pts)
                sm = soft_min(areas, 100)
                try:
                    hull = ConvexHull(pts)
                    hull_area = hull.volume
                    if hull_area < 1e-10:
                        return 1e6
                except:
                    return 1e6
                score = sm / hull_area
                return -score
            except:
                return 1e6
        
        # Bounds: points in [-2, 2]^2
        bounds = [(-2, 2)] * (N * 2)
        
        try:
            de_res = differential_evolution(de_objective, bounds, maxiter=50, 
                                           popsize=10, tol=1e-6, seed=42)
            if de_res.fun < 1e5:
                pts = de_res.x.reshape(N, 2)
                update_best(pts)
        except:
            pass
    
    # Strategy 4: Local refinement with small perturbations
    if best_points is not None and time_left() > 15.0:
        x_best = best_points.flatten().copy()
        
        for pert_scale in [0.01, 0.05]:
            if time_left() < 10.0:
                break
            for _ in range(5):
                if time_left() < 5.0:
                    break
                x_pert = x_best + np.random.normal(0, pert_scale, len(x_best))
                
                try:
                    res_p = minimize(lambda p: objective_2d(p, 200, 'general'), x_pert,
                                   method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                    if res_p.fun < 1e5:
                        pts = res_p.x.reshape(N, 2)
                        s = get_score(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts.copy()
                            x_best = res_p.x
                except:
                    pass
    
    # Final fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = angles_to_points(angles)
        best_score = get_score(best_points)
    
    return best_points
