"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial import ConvexHull
import itertools


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_tri_area:
                    min_tri_area = area
    return min_tri_area / hull_area


def _negative_score(points_flat):
    """Negative of the score for minimization."""
    points = points_flat.reshape(N, 2)
    return -_score(points)


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _concentric_rings(n):
    """Generate points on concentric rings."""
    points = []
    # Center
    points.append([0.0, 0.0])
    # First ring
    n1 = 5
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    r1 = 0.5
    points.extend([[r1 * np.cos(a), r1 * np.sin(a)] for a in angles1])
    # Second ring
    n2 = 8
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    r2 = 1.0
    points.extend([[r2 * np.cos(a), r2 * np.sin(a)] for a in angles2])
    return np.array(points[:n])


def construct():
    deadline = time.time() + 110  # Leave 40s margin
    
    best_points = None
    best_score_val = 0.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular polygon
    initial_configs.append(_regular_polygon(N, 1.0).flatten())
    
    # 2. Concentric rings
    initial_configs.append(_concentric_rings(N).flatten())
    
    # 3. Grid-like
    grid_points = []
    for i in range(4):
        for j in range(4):
            grid_points.append([i - 1.5, j - 1.5])
    initial_configs.append(np.array(grid_points[:N]).flatten())
    
    # 4. Random with different seeds
    for seed in [42, 123, 456, 789]:
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-1, 1, (N, 2))
        initial_configs.append(pts.flatten())
    
    # 5. Jittered regular polygon
    for seed in [11, 22]:
        rng = np.random.RandomState(seed)
        base = _regular_polygon(N, 1.0)
        jittered = base + rng.normal(0, 0.1, (N, 2))
        initial_configs.append(jittered.flatten())
    
    # Try each initial configuration with Nelder-Mead
    for idx, x0 in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _negative_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result.success or result.fun < 0:
                score_val = -result.fun
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_points = result.x.reshape(N, 2).copy()
            
            # Also check the initial point
            init_score = -_negative_score(x0)
            if init_score > best_score_val:
                best_score_val = init_score
                best_points = x0.reshape(N, 2).copy()
                
        except Exception:
            continue
    
    # If no good points found, fall back to regular polygon
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    # Normalize: scale so that convex hull area = 1
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-12:
        scale = 1.0 / np.sqrt(hull_area)
        best_points = best_points * scale
    
    return best_points
# EVOLVE-BLOCK-END
