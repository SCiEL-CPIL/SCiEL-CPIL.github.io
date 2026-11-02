"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _triangle_area(p1, p2, p3):
    """Area of triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _score(points):
    """Compute min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    
    return min_area / hull_area


def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with regular 14-gon inscribed in unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = x0.copy()
    best_score = _score(best_points)
    
    # Try multiple initializations and local optimizations
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Different starting configurations
        if attempts == 1:
            # Regular 14-gon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            x0 = np.column_stack([np.cos(angles), np.sin(angles)])
        elif attempts == 2:
            # Regular 14-gon with slight perturbation
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.05 * np.random.randn(N)
            angles = np.sort(angles)
            x0 = np.column_stack([np.cos(angles), np.sin(angles)])
        elif attempts == 3:
            # Two concentric rings: 7 points on inner, 7 on outer
            r_inner = 0.5
            r_outer = 1.0
            angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
            x0 = np.vstack([
                r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)]),
                r_outer * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            ])
        elif attempts == 4:
            # Regular 14-gon with different phase
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.1
            x0 = np.column_stack([np.cos(angles), np.sin(angles)])
        elif attempts == 5:
            # Grid-like arrangement in convex hull
            # 4x4 grid minus 2 corners, then optimized
            xs = np.linspace(-1, 1, 4)
            ys = np.linspace(-1, 1, 4)
            grid = np.array([[x, y] for x in xs for y in ys])
            # Take 14 points from the 16-point grid
            x0 = grid[:14].copy()
        else:
            # Random perturbation of best found so far
            x0 = best_points + 0.1 * np.random.randn(N, 2)
        
        # Local optimization using Nelder-Mead
        try:
            result = minimize(
                _neg_score,
                x0.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_points = result.x.reshape(-1, 2)
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass
        
        # Check if we should continue
        if time.time() > deadline:
            break
    
    return best_points
# EVOLVE-BLOCK-END
