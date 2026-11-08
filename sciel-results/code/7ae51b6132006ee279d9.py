"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 13

# EVOLVE-BLOCK-START
import random
import itertools
from scipy.spatial import ConvexHull

def _triangle_areas(points):
    """Compute all triangle areas for N points. Returns array of size C(N,3)."""
    n = len(points)
    areas = []
    for i, j, k in itertools.combinations(range(n), 3):
        a = points[i]
        b = points[j]
        c = points[k]
        area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
        areas.append(area)
    return np.array(areas)

def _convex_hull_area(points):
    """Compute convex hull area of the point set."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback: use shoelace on sorted points
        # This is a rough fallback
        from scipy.spatial import ConvexHull as CH
        try:
            h = CH(points)
            return h.volume
        except:
            # Very rough: bounding box area
            return (points[:,0].max()-points[:,0].min()) * (points[:,1].max()-points[:,1].min())

def _score(points):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _initial_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. Regular 13-gon (centered)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 1.0
    pts = np.column_stack([r*np.cos(angles), r*np.sin(angles)])
    candidates.append(pts)
    
    # 2. Grid-like arrangement
    # 4x4 grid minus one corner, then perturb
    grid = []
    for i in range(4):
        for j in range(4):
            grid.append([i, j])
    # Remove one to get 13
    grid = grid[:13]
    grid = np.array(grid, dtype=float)
    grid = (grid - grid.mean(axis=0)) / (grid.max(axis=0) - grid.min(axis=0) + 1e-10)
    candidates.append(grid)
    
    # 3. Random with fixed seed
    rng = np.random.RandomState(42)
    pts = rng.rand(N, 2)
    candidates.append(pts)
    
    # 4. Another random
    rng = np.random.RandomState(123)
    pts = rng.rand(N, 2)
    candidates.append(pts)
    
    # 5. Two concentric rings
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    candidates.append(pts)
    
    # 6. Perturbed regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 1.0 + 0.1*np.sin(2*angles)
    pts = np.column_stack([r*np.cos(angles), r*np.sin(angles)])
    candidates.append(pts)
    
    return candidates

def _local_optimize(points, time_limit):
    """Local optimization using Nelder-Mead on flattened point coordinates."""
    best_points = points.copy()
    best_score = _score(best_points)
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalize if points are too close or hull area is tiny
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-10:
            return 1e10
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        return -min_area / hull_area  # Minimize negative score
    
    x0 = best_points.flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.x is not None:
            new_points = result.x.reshape(-1, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
    except Exception:
        pass
    
    return best_points, best_score

def _mutate(points, rng, strength=0.1):
    """Random mutation of points."""
    mutated = points.copy()
    for i in range(len(mutated)):
        mutated[i] += rng.normal(0, strength, size=2)
    return mutated

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Generate and evaluate initial candidates
    candidates = _initial_candidates()
    for cand in candidates:
        if time.time() > deadline:
            break
        score = _score(cand)
        if score > best_score:
            best_score = score
            best_points = cand.copy()
    
    # Local optimization on best candidate
    if best_points is not None:
        if time.time() < deadline - 5:
            best_points, best_score = _local_optimize(best_points, deadline)
    
    # Try local optimization on other candidates if time permits
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        if np.allclose(cand, best_points, atol=1e-6):
            continue
        try:
            if time.time() > deadline - 10:
                break
            opt_points, opt_score = _local_optimize(cand, deadline)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
        except Exception:
            continue
    
    # Random restarts with local optimization if time permits
    rng = np.random.RandomState(999)
    attempt = 0
    while time.time() < deadline - 15:
        attempt += 1
        # Random initial point
        pts = rng.rand(N, 2)
        # Quick local optimization with limited iterations
        try:
            def objective(x):
                p = x.reshape(-1, 2)
                hull_area = _convex_hull_area(p)
                if hull_area < 1e-10:
                    return 1e10
                areas = _triangle_areas(p)
                min_area = np.min(areas)
                return -min_area / hull_area
            
            x0 = pts.flatten()
            result = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.x is not None:
                new_points = result.x.reshape(-1, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            continue
    
    # Final check
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
