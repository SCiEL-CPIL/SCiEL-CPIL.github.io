# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def compute_score(points):
    """Compute the Heilbronn score: A_min / A_hull (vectorized)"""
    n = len(points)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area computation
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / A_hull

def objective(points_flat):
    """Objective function for minimization (we minimize negative score)"""
    points = points_flat.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    score = compute_score(points)
    if score <= 0:
        return 1e10
    return -score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    Uses Nelder-Mead optimization from diverse structured starts.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s limit
    
    def time_left():
        return deadline - time.time()
    
    # Initial configurations based on known Heilbronn structures
    
    # Config 1: Regular 12-gon + center
    def make_config1():
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([outer, center])
    
    # Config 2: Two concentric rings (7 outer, 6 inner)
    def make_config2():
        angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        outer = np.column_stack([np.cos(angles1), np.sin(angles1)])
        inner = np.column_stack([0.55 * np.cos(angles2), 0.55 * np.sin(angles2)])
        return np.vstack([outer, inner])
    
    # Config 3: Three rings (5, 4, 4)
    def make_config3():
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
        p3 = np.column_stack([0.3 * np.cos(a3), 0.3 * np.sin(a3)])
        return np.vstack([p1, p2, p3])
    
    # Config 4: 1 center, 6 at r=0.5, 6 at r=1.0
    def make_config4():
        a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.5 * np.cos(a1), 0.5 * np.sin(a1)])
        p3 = np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([p1, p2, p3])
    
    # Config 5: Perturbed regular 13-gon
    def make_config5():
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        rng = np.random.default_rng(42)
        points += rng.normal(0, 0.03, size=points.shape)
        return points
    
    # Config 6: 1 center, 3 at r=0.3, 3 at r=0.6, 6 at r=1.0
    def make_config6():
        a1 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
        a3 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.3 * np.cos(a1), 0.3 * np.sin(a1)])
        p3 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
        p4 = np.column_stack([np.cos(a3), np.sin(a3)])
        return np.vstack([p1, p2, p3, p4])
    
    # Config 7: Two rings with different radii (6 outer, 6 inner, 1 center)
    def make_config7():
        a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.4 * np.cos(a1), 0.4 * np.sin(a1)])
        p3 = np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([p1, p2, p3])
    
    # Config 8: Three rings (4, 4, 5)
    def make_config8():
        a1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([0.55 * np.cos(a2), 0.55 * np.sin(a2)])
        p3 = np.column_stack([0.25 * np.cos(a3), 0.25 * np.sin(a3)])
        return np.vstack([p1, p2, p3])
    
    initial_configs = [
        make_config1(),
        make_config2(),
        make_config3(),
        make_config4(),
        make_config5(),
        make_config6(),
        make_config7(),
        make_config8(),
    ]
    
    # Add some random starts
    rng = np.random.default_rng(7)
    for _ in range(4):
        pts = rng.uniform(-1, 1, size=(13, 2))
        initial_configs.append(pts)
    
    best_points = None
    best_score = 0.0
    
    # Nelder-Mead options
    nm_options = {
        'maxiter': 3000,
        'xatol': 1e-6,
        'fatol': 1e-8,
        'adaptive': True,
    }
    
    for config in initial_configs:
        if time_left() < 15:
            break
        
        points = config.copy()
        score = compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Run Nelder-Mead optimization
        x0 = points.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options=nm_options
            )
            
            if result.success or result.fun < 0:
                opt_points = result.x.reshape(13, 2)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    # Second pass: refine best result with more iterations
    if time_left() > 60 and best_points is not None:
        x0 = best_points.flatten()
        nm_options2 = {
            'maxiter': 5000,
            'xatol': 1e-7,
            'fatol': 1e-9,
            'adaptive': True,
        }
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options=nm_options2
            )
            
            if result.success or result.fun < 0:
                opt_points = result.x.reshape(13, 2)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = make_config1()
    
    # Final validation
    try:
        hull = ConvexHull(best_points)
    except Exception:
        best_points = make_config1()
    
    return best_points

# EVOLVE-BLOCK-END
