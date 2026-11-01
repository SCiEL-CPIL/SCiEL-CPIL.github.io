import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                cross = (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - \
                        (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                areas[i, j, k] = 0.5 * abs(cross)
                areas[j, i, k] = areas[i, j, k]
                areas[k, j, i] = areas[i, j, k]
    return areas

def _min_triangle_area(pts):
    """Compute the minimum triangle area over all triples."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - \
                        (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the point set."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_score = 0.0
    
    # Fix p0 at (0,0) and p1 at (1,0) to remove translation/rotation invariance
    # Variables: 2*(N-2) = 24 coordinates for p2...p13
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def objective(x):
        pts = get_points(x)
        return -_score(pts)
    
    # Initial guess: regular polygon-like configuration
    def initial_guess():
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        # Normalize so that p0=(0,0) and p1=(1,0)
        # Translate so p0 is at origin
        pts = pts - pts[0]
        # Rotate so p1 is along x-axis
        p1 = pts[1]
        angle = np.arctan2(p1[1], p1[0])
        rot = np.array([[np.cos(-angle), -np.sin(-angle)],
                        [np.sin(-angle), np.cos(-angle)]])
        pts = pts @ rot.T
        # Scale so p1 is at (1,0)
        scale = np.linalg.norm(pts[1])
        if scale > 1e-12:
            pts = pts / scale
        # Extract free variables
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            x[2 * i] = pts[i + 2, 0]
            x[2 * i + 1] = pts[i + 2, 1]
        return x
    
    x0 = initial_guess()
    
    # Try a few different initial configurations
    configs = [x0]
    
    # Add some perturbed versions
    rng = np.random.RandomState(42)
    for _ in range(5):
        x_perturbed = x0 + rng.randn(2 * (N - 2)) * 0.5
        configs.append(x_perturbed)
    
    bounds = [(-5, 5)] * (2 * (N - 2))
    
    # Try each configuration with local refinement
    for x_init in configs:
        if time.time() > deadline:
            break
        
        # First try L-BFGS-B local optimization
        try:
            result = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-12})
            if time.time() > deadline:
                break
            
            x_best = result.x
            pts_best = get_points(x_best)
            score_best = _score(pts_best)
            
            if score_best > best_score:
                best_score = score_best
                best_pts = pts_best.copy()
        except Exception:
            pass
    
    # If time remains, try differential evolution for global search
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                objective, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-6, polish=True, init='latinhypercube'
            )
            if time.time() > deadline:
                break
            
            x_de = de_result.x
            pts_de = get_points(x_de)
            score_de = _score(pts_de)
            
            if score_de > best_score:
                best_score = score_de
                best_pts = pts_de.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if best_pts is not None and time.time() < deadline - 5:
        x_final = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            x_final[2 * i] = best_pts[i + 2, 0]
            x_final[2 * i + 1] = best_pts[i + 2, 1]
        
        try:
            result = minimize(objective, x_final, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            if time.time() > deadline:
                break
            
            x_polished = result.x
            pts_polished = get_points(x_polished)
            score_polished = _score(pts_polished)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_pts
