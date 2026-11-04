"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross product|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute hull manually or return a default
        # For robustness, if ConvexHull fails, we might need a different approach
        # But for N=14, ConvexHull should be robust
        return 1.0

def _score(points):
    """Score is the minimum triangle area divided by the convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri_area = _min_triangle_area(points)
    return min_tri_area / hull_area

def _objective(params):
    """Objective function for optimization: negative score (to minimize)."""
    points = params.reshape(N, 2)
    # Ensure points are well-distributed (avoid degeneracy)
    # Add a small penalty if points are too close or hull area is too small
    score = _score(points)
    return -score

def construct():
    deadline = time.time() + 120  # 120 seconds total
    best_score = 0.0
    best_points = None
    
    # Strategy: Use symmetry-constrained ring parameters
    # For N=14, we can try placing points on two concentric rings
    # Outer ring: 7 points evenly spaced
    # Inner ring: 7 points evenly spaced, rotated by half-step
    # This reduces the search space significantly
    
    # Parameters:
    # r1: radius of outer ring
    # r2: radius of inner ring
    # theta: rotation offset of inner ring relative to outer
    # We can also vary the number of points on each ring, but let's start with 7-7 split
    
    # Actually, let's use a more general approach:
    # Place points on a circle with some perturbation, then optimize
    
    # Initial guess: points on a unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_params = np.array([[np.cos(a), np.sin(a)] for a in angles]).flatten()
    
    # Try multiple initializations
    initializations = [initial_params]
    
    # Also try a two-ring configuration
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    r_outer = 1.0
    r_inner = 0.5
    two_ring_params = np.array([
        [r_outer * np.cos(a), r_outer * np.sin(a)] for a in angles_outer
    ] + [
        [r_inner * np.cos(a), r_inner * np.sin(a)] for a in angles_inner
    ]).flatten()
    initializations.append(two_ring_params)
    
    # Add a random initialization
    rng = np.random.RandomState(42)
    random_params = rng.randn(N * 2) * 0.5
    initializations.append(random_params)
    
    for init in initializations:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(_objective, init, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.success or result.fun < 0:
                points = result.x.reshape(N, 2)
                score = _score(points)
                
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
        except Exception as e:
            continue
    
    # If no valid points found, return a default configuration
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    return best_points
# EVOLVE-BLOCK-END
