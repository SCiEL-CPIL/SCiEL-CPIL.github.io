import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Pre-compute all triangle index combinations
    tri_indices = np.array(list(combinations(range(N), 3)))
    
    def compute_tri_areas(pts):
        """Compute all triangle areas for N points."""
        pi = pts[tri_indices[:, 0]]
        pj = pts[tri_indices[:, 1]]
        pk = pts[tri_indices[:, 2]]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return areas
    
    def compute_min_tri_area(pts):
        """Compute minimum triangle area for N points."""
        return np.min(compute_tri_areas(pts))
    
    def hull_area(pts):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1e-10
    
    def score(pts):
        """Score = min triangle area / hull area."""
        h_area = hull_area(pts)
        if h_area < 1e-10:
            return 0.0
        m_area = compute_min_tri_area(pts)
        return m_area / h_area
    
    def soft_min_area(pts, beta=50.0):
        """Soft-min surrogate using log-sum-exp for smoothness."""
        areas = compute_tri_areas(pts)
        # Soft min: -1/beta * log(sum(exp(-beta * areas)))
        # This is smooth and approximates min(areas) as beta -> inf
        min_val = np.min(areas)
        shifted = areas - min_val
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        return min_val - (1.0/beta) * log_sum
    
    def score_smooth(pts, beta=50.0):
        """Smooth score using soft-min."""
        h_area = hull_area(pts)
        if h_area < 1e-10:
            return 0.0
        m_area = soft_min_area(pts, beta)
        return m_area / h_area
    
    def neg_score_angular(angles):
        """Negative score for angular parameterization on unit circle."""
        if time.time() > deadline:
            return 0.0
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return -score(pts)
    
    def neg_score_smooth_angular(angles, beta=50.0):
        """Negative smooth score for angular parameterization."""
        if time.time() > deadline:
            return 0.0
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return -score_smooth(pts, beta)
    
    def neg_score_free(flat_pts):
        """Negative score for free 2D points."""
        if time.time() > deadline:
            return 0.0
        pts = flat_pts.reshape(N, 2)
        return -score(pts)
    
    def neg_score_smooth_free(flat_pts, beta=50.0):
        """Negative smooth score for free 2D points."""
        if time.time() > deadline:
            return 0.0
        pts = flat_pts.reshape(N, 2)
        return -score_smooth(pts, beta)
    
    best_score_val = 0.0
    best_result = None
    
    # Strategy 1: Optimize angular positions on unit circle with smooth objective
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    rng = np.random.default_rng(42)
    
    # Phase 1a: Smooth optimization with increasing beta (annealing)
    for beta in [10.0, 30.0, 50.0, 100.0]:
        if time.time() > deadline - 20:
            break
        
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            if trial == 0:
                angles0 = base_angles.copy()
            else:
                angles0 = base_angles + rng.normal(0, 0.03, N)
                angles0 = np.sort(angles0) % (2*np.pi)
            
            try:
                res = minimize(lambda a: neg_score_smooth_angular(a, beta), angles0,
                              method='Powell',
                              options={'maxiter': 300, 'ftol': 1e-12, 'disp': False})
                if res.fun < 0:
                    pts = np.column_stack([np.cos(res.x), np.sin(res.x)])
                    s = score(pts)
                    if s > best_score_val:
                        best_score_val = s
                        best_result = pts
            except Exception:
                pass
    
    # Phase 1b: Exact (nonsmooth) optimization from best smooth result
    if best_result is not None and time.time() < deadline - 15:
        # Extract angles from best result
        best_angles = np.arctan2(best_result[:, 1], best_result[:, 0])
        best_angles = np.sort(best_angles) % (2*np.pi)
        
        for trial in range(4):
            if time.time() > deadline - 5:
                break
            
            if trial == 0:
                angles0 = best_angles.copy()
            else:
                angles0 = best_angles + rng.normal(0, 0.02, N)
                angles0 = np.sort(angles0) % (2*np.pi)
            
            try:
                res = minimize(neg_score_angular, angles0, method='Powell',
                              options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
                if res.fun < 0:
                    pts = np.column_stack([np.cos(res.x), np.sin(res.x)])
                    s = score(pts)
                    if s > best_score_val:
                        best_score_val = s
                        best_result = pts
            except Exception:
                pass
    
    # Phase 2: Free 2D optimization with smooth objective
    if best_result is not None and time.time() < deadline - 15:
        x0_free = best_result.flatten()
        
        for beta in [30.0, 100.0]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(lambda f: neg_score_smooth_free(f, beta), x0_free,
                              method='Powell',
                              options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
                if res.fun < 0:
                    pts = res.x.reshape(N, 2)
                    s = score(pts)
                    if s > best_score_val:
                        best_score_val = s
                        best_result = pts
                    x0_free = pts.flatten()
            except Exception:
                pass
        
        # Final exact polish in free 2D
        if time.time() < deadline - 5:
            try:
                res = minimize(neg_score_free, x0_free, method='Powell',
                              options={'maxiter': 300, 'ftol': 1e-14, 'disp': False})
                if res.fun < 0:
                    pts = res.x.reshape(N, 2)
                    s = score(pts)
                    if s > best_score_val:
                        best_score_val = s
                        best_result = pts
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead with very tight tolerance for final polish
    if best_result is not None and time.time() < deadline - 5:
        # Work in angular space for final polish
        best_angles = np.arctan2(best_result[:, 1], best_result[:, 0])
        best_angles = np.sort(best_angles) % (2*np.pi)
        
        try:
            res = minimize(neg_score_angular, best_angles, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'disp': False})
            if res.fun < 0:
                pts = np.column_stack([np.cos(res.x), np.sin(res.x)])
                s = score(pts)
                if s > best_score_val:
                    best_score_val = s
                    best_result = pts
        except Exception:
            pass
    
    # Phase 4: Final exact polish in free 2D with Nelder-Mead
    if best_result is not None and time.time() < deadline - 3:
        x0 = best_result.flatten()
        try:
            res = minimize(neg_score_free, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14, 'disp': False})
            if res.fun < 0:
                pts = res.x.reshape(N, 2)
                s = score(pts)
                if s > best_score_val:
                    best_score_val = s
                    best_result = pts
        except Exception:
            pass
    
    # Final: if no result, return regular polygon
    if best_result is None:
        best_result = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    return best_result
