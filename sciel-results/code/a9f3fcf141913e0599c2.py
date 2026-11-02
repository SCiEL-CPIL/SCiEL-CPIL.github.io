"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(points):
    """Compute minimum triangle area / convex hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    # Use a simple convex hull algorithm (Andrew's monotone chain)
    pts = points.copy()
    pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    
    # Convex hull area using shoelace formula
    hull = hull.reshape(-1, 2)
    x = hull[:, 0]
    y = hull[:, 1]
    hull_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    
    if hull_area <= 0:
        return 0.0
    
    # Compute minimum triangle area
    min_tri_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_tri_area:
                    min_tri_area = area
    
    return min_tri_area / hull_area


def _neg_score(points):
    return -_score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with regular polygon as initial guess
    def regular_polygon(n, center=(0, 0), radius=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([center[0] + radius * np.cos(angles),
                                 center[1] + radius * np.sin(angles)])
    
    best_points = regular_polygon(N)
    best_score = _score(best_points)
    
    # Try multiple random initializations and local optimization
    rng = np.random.RandomState(42)
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Generate random initial points
        if attempt == 0:
            x0 = regular_polygon(N)
        else:
            x0 = regular_polygon(N) + rng.randn(N, 2) * 0.3
        
        # Flatten to 1D for scipy minimize
        x0_flat = x0.flatten()
        
        def objective(flat_x):
            pts = flat_x.reshape(-1, 2)
            return -_score(pts)
        
        # Use Nelder-Mead for local optimization
        try:
            res = minimize(objective, x0_flat, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < 0:
                trial_points = res.x.reshape(-1, 2)
                trial_score = _score(trial_points)
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = trial_points.copy()
        except Exception:
            pass
    
    # Final polish with more iterations if time permits
    if time.time() < deadline - 5:
        x0_flat = best_points.flatten()
        
        def objective(flat_x):
            pts = flat_x.reshape(-1, 2)
            return -_score(pts)
        
        try:
            res = minimize(objective, x0_flat, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            
            if res.fun < 0:
                trial_points = res.x.reshape(-1, 2)
                trial_score = _score(trial_points)
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = trial_points.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
