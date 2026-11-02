import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if all points are collinear or fewer than 3
        return 0.0

def _score(pts):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _negative_score(pts):
    """Objective function for minimization (we want to maximize score)."""
    return -_score(pts)

def _initial_fermat(n, seed=0):
    """Generate initial points using Fermat spiral (Vogel's model)."""
    i = np.arange(n)
    r = np.sqrt(i / n)
    theta = i * 2.399963229728653  # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    pts -= np.mean(pts, axis=0)
    return pts

def _initial_regular_polygon(n, seed=0):
    """Generate initial points on a regular polygon with some jitter."""
    angles = 2 * np.pi * np.arange(n) / n
    r = np.ones(n)
    rng = np.random.RandomState(seed)
    r = 1.0 + 0.1 * rng.randn(n)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    pts = np.column_stack((x, y))
    pts -= np.mean(pts, axis=0)
    return pts

def _initial_random(n, seed=0):
    """Generate random initial points in a unit square."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(n, 2)
    pts -= np.mean(pts, axis=0)
    return pts

def _initial_grid(n, seed=0):
    """Generate initial points on a grid-like structure."""
    rng = np.random.RandomState(seed)
    # Create a grid of points
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count < n:
                x = i - side / 2.0
                y = j - side / 2.0
                # Add small random perturbation
                x += 0.05 * rng.randn()
                y += 0.05 * rng.randn()
                pts.append([x, y])
                count += 1
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    best_score = -np.inf
    best_pts = None
    
    # Generate multiple initial configurations
    initializers = [
        ("fermat", _initial_fermat),
        ("polygon", _initial_regular_polygon),
        ("random", _initial_random),
        ("grid", _initial_grid),
    ]
    
    # Try multiple random seeds for each initializer
    seeds = range(5)
    
    all_initials = []
    for name, init_func in initializers:
        for seed in seeds:
            pts = init_func(N, seed)
            score = _score(pts)
            all_initials.append((pts, score, f"{name}_{seed}"))
    
    # Sort by initial score (descending) and take top candidates
    all_initials.sort(key=lambda x: x[1], reverse=True)
    
    # Optimize top candidates
    max_candidates = min(10, len(all_initials))
    
    for idx in range(max_candidates):
        if time.time() > deadline - 15.0:
            break
        
        pts0, init_score, name = all_initials[idx]
        
        # Flatten points for optimization
        x0 = pts0.flatten()
        
        # Use Nelder-Mead for optimization
        try:
            result = minimize(
                _negative_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < -init_score:
                opt_pts = result.x.reshape(-1, 2)
                opt_score = _score(opt_pts)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts
                    
        except Exception:
            continue
    
    # If no optimization succeeded, use best initial
    if best_pts is None:
        best_pts = all_initials[0][0]
        best_score = all_initials[0][1]
    
    # Final check: ensure we have valid points
    if best_pts is None or len(best_pts) != N:
        # Fallback to random points
        rng = np.random.RandomState(N)
        best_pts = rng.rand(N, 2)
    
    return best_pts

# EVOLVE-BLOCK-END
