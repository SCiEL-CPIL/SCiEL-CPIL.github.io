import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

def _min_triangle_area(points):
    """Compute minimum triangle area for N points (N<=14)."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            v1 = points[j] - points[i]
            for k in range(j+1, n):
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute min triangle area / convex hull area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _make_angular_ring(n, radius=1.0, phase=0.0):
    """Create n points on a circle of given radius."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _construct_two_ring(n_out, r_in, phase_in):
    """Construct two-ring configuration."""
    outer = _make_angular_ring(n_out, 1.0, 0.0)
    if n_out == N:
        return outer
    inner = _make_angular_ring(N - n_out, r_in, phase_in)
    return np.vstack((outer, inner))

def _optimize_two_ring(n_out, time_budget):
    """Optimize two-ring configuration with n_out outer points."""
    n_in = N - n_out
    if n_in == 0:
        pts = _make_angular_ring(N, 1.0, 0.0)
        return _score(pts), pts
    
    def objective(params):
        r_in, phase_in = params
        if r_in < 0.01 or r_in > 1.4:
            return 1e6
        pts = _construct_two_ring(n_out, r_in, phase_in)
        return _neg_score(pts)
    
    # Use differential evolution for global search
    bounds = [(0.05, 1.3), (0, 2*np.pi)]
    try:
        result = differential_evolution(
            objective, bounds, 
            maxiter=50, popsize=15,
            tol=1e-6, seed=42,
            polish=True
        )
        best_params = result.x
        best_score = -result.fun
    except Exception:
        # Fallback: grid search
        best_score = 0
        best_params = [0.5, 0.0]
        for r in np.linspace(0.1, 1.2, 12):
            for ph in np.linspace(0, 2*np.pi, 12, endpoint=False):
                s = -objective([r, ph])
                if s > best_score:
                    best_score = s
                    best_params = [r, ph]
    
    # Local polish with Nelder-Mead
    try:
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':200})
        if -res.fun > best_score:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    pts = _construct_two_ring(n_out, best_params[0], best_params[1])
    return best_score, pts

def _polish_full(points, time_budget):
    """Polish full 28D configuration."""
    def obj_flat(x):
        p = x.reshape(N, 2)
        return _neg_score(p)
    
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(obj_flat, points.flatten(), method='Nelder-Mead',
                       options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':300})
        pts = res.x.reshape(N, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    except Exception:
        pass
    
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Try various two-ring configurations
    splits = [(14,0), (12,2), (10,4), (8,6), (7,7), (11,3), (9,5), (13,1)]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        
        time_budget = min(15.0, remaining / 2.0)
        score, pts = _optimize_two_ring(n_out, time_budget)
        try_update(score, pts)
    
    # Also try a three-ring configuration for more flexibility
    if time.time() < deadline - 25:
        def _three_ring(r1, r2, ph1, ph2):
            # 6 outer, 5 middle, 3 inner
            outer = _make_angular_ring(6, 1.0, 0.0)
            middle = _make_angular_ring(5, r1, ph1)
            inner = _make_angular_ring(3, r2, ph2)
            return np.vstack((outer, middle, inner))
        
        def obj_three(params):
            r1, r2, ph1, ph2 = params
            if r1 < 0.05 or r1 > 1.3 or r2 < 0.05 or r2 > 1.0:
                return 1e6
            pts = _three_ring(r1, r2, ph1, ph2)
            return _neg_score(pts)
        
        bounds = [(0.1, 1.2), (0.05, 0.9), (0, 2*np.pi), (0, 2*np.pi)]
        try:
            result = differential_evolution(
                obj_three, bounds,
                maxiter=30, popsize=12,
                tol=1e-5, seed=7,
                polish=False
            )
            params = result.x
            pts = _three_ring(params[0], params[1], params[2], params[3])
            s = _score(pts)
            try_update(s, pts)
            
            # Polish
            if time.time() < deadline - 10:
                s2, pts2 = _polish_full(pts, 10)
                try_update(s2, pts2)
        except Exception:
            pass
    
    # Final polish on best found
    if best_points is not None and time.time() < deadline - 5:
        s, pts = _polish_full(best_points, deadline - time.time() - 5)
        try_update(s, pts)
    
    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
