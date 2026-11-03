"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Use Andrew's monotone chain
    pts = sorted(points.tolist())
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Concatenate to form convex hull
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
    
    # Compute area of convex polygon
    area = 0.0
    m = len(hull)
    for i in range(m):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % m]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _generate_points(ratio, phase):
    """
    Generate 13 points: 1 center + 12 outer ring.
    ratio: radial scale of outer ring relative to unit circle
    phase: rotational offset of the outer ring (radians)
    """
    center = np.array([0.0, 0.0])
    outer_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + phase
    outer_pts = np.column_stack([
        ratio * np.cos(outer_angles),
        ratio * np.sin(outer_angles)
    ])
    points = np.vstack([center, outer_pts])
    return points


def _objective(params):
    """Objective to minimize: negative of score."""
    ratio, phase = params
    points = _generate_points(ratio, phase)
    return -_score(points)


def construct():
    # Grid search over ratio and phase
    best_score = -1.0
    best_params = (1.0, 0.0)
    
    # Ratio range: outer ring radius. Since we have a center point, 
    # the convex hull is determined by the outer ring.
    # Try a range of ratios and phases.
    ratios = np.linspace(0.3, 1.5, 50)
    phases = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    
    for r in ratios:
        for ph in phases:
            points = _generate_points(r, ph)
            s = _score(points)
            if s > best_score:
                best_score = s
                best_params = (r, ph)
    
    # Newton-Raphson / gradient-free refinement using scipy minimize
    # Use Nelder-Mead for robustness on the 2D manifold
    result = minimize(_objective, best_params, method='Nelder-Mead', 
                      options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 10000})
    
    if result.success or result.fun < _objective(best_params):
        best_params = result.x
    
    # Final fine-tuning with a local grid around the best found
    r0, ph0 = best_params
    fine_ratios = np.linspace(r0 - 0.05, r0 + 0.05, 30)
    fine_phases = np.linspace(ph0 - 0.05, ph0 + 0.05, 30)
    
    for r in fine_ratios:
        for ph in fine_phases:
            points = _generate_points(r, ph)
            s = _score(points)
            if s > best_score:
                best_score = s
                best_params = (r, ph)
    
    # One more refinement
    result2 = minimize(_objective, best_params, method='Nelder-Mead', 
                       options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 5000})
    if result2.success or result2.fun < _objective(best_params):
        best_params = result2.x
    
    final_points = _generate_points(*best_params)
    return final_points
# EVOLVE-BLOCK-END
