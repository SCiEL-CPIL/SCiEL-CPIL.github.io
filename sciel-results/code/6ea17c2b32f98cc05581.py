"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Find the point with lowest y (and lowest x in case of tie)
    start = 0
    for i in range(1, n):
        if points[i][1] < points[start][1] or (points[i][1] == points[start][1] and points[i][0] < points[start][0]):
            start = i
    
    # Sort by polar angle
    def angle_cmp(p):
        return np.arctan2(p[1] - points[start][1], p[0] - points[start][0])
    
    sorted_pts = sorted(points, key=angle_cmp)
    sorted_pts.insert(0, points[start])  # Put start first
    
    # Andrew's monotone chain algorithm
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    normalized by the convex hull area.
    
    New Approach: Differential Evolution with Gauge Fixing.
    1. Fix P0=(0,0) and P1=(1,0) to remove translation and rotation invariance.
    2. Optimize the remaining 24 coordinates using Differential Evolution.
    3. This is a global optimization method, distinct from the local Nelder-Mead used previously.
    """
    
    # Gauge fixing: P0 and P1 are fixed.
    # Variables: x[0]...x[23] correspond to P2...P13 coordinates.
    # P0 = (0, 0)
    # P1 = (1, 0)
    
    def objective(x):
        # Reconstruct points
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        # x is flattened array of shape (24,)
        # P2 is x[0], x[1]
        # P3 is x[2], x[3]
        # ...
        pts[2:] = x.reshape(N - 2, 2)
        
        # Check for degeneracy (collinear points)
        # If any triangle area is 0, score is 0.
        # We want to maximize score, so minimize -score.
        s = _score(pts)
        return -s

    # Bounds for the 24 variables
    # We allow points to be in a box [-3, 3] relative to the fixed points.
    # This is large enough to form any convex shape of reasonable aspect ratio.
    bounds = [(-3.0, 3.0)] * (2 * (N - 2))
    
    # Run Differential Evolution
    # popsize: 15 * dim is standard, maybe 10 * dim for speed.
    # maxiter: 200-300 should be sufficient for 24 dims.
    # tol: 1e-4
    # seed: Fixed for reproducibility, but DE is stochastic.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=300, 
            tol=1e-4, 
            polish=True,  # Use L-BFGS for final polish
            init='sobol'  # Sobol sampling for better coverage
        )
        
        best_x = result.x
        best_score_val = -result.fun
        
        # Construct the final points
        best_pts = np.zeros((N, 2))
        best_pts[0] = [0.0, 0.0]
        best_pts[1] = [1.0, 0.0]
        best_pts[2:] = best_x.reshape(N - 2, 2)
        
        # Verify score
        final_score = _score(best_pts)
        
        return best_pts

    except Exception:
        # Fallback to regular polygon if DE fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
# EVOLVE-BLOCK-END
