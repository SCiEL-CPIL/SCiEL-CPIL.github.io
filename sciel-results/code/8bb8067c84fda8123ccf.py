"""Heilbronn convex regions - multi-shell regular polygon optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _convex_hull_area(points):
    """Area of convex hull of points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, 'volume' is area
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _generate_points(m_outer, k_inner, r_ratio, rot_angle):
    """Generate points: m_outer on unit circle, k_inner on circle of radius r_ratio, rotated by rot_angle."""
    points = []
    if m_outer > 0:
        angles_outer = 2 * np.pi * np.arange(m_outer) / m_outer
        for a in angles_outer:
            points.append([np.cos(a), np.sin(a)])
    if k_inner > 0:
        angles_inner = 2 * np.pi * np.arange(k_inner) / k_inner + rot_angle
        for a in angles_inner:
            points.append([r_ratio * np.cos(a), r_ratio * np.sin(a)])
    return np.array(points)

def _objective(params, m_outer, k_inner):
    """Negative score for minimization."""
    r_ratio, rot_angle = params
    # Clamp to valid range
    r_ratio = max(0.01, min(0.99, r_ratio))
    points = _generate_points(m_outer, k_inner, r_ratio, rot_angle)
    return -_score(points)

def construct():
    deadline = time.time() + 120.0
    safety_margin = 10.0  # seconds before deadline
    
    best_score = 0.0
    best_points = None
    
    # Try different partitions of 13 points into two shells
    partitions = [
        (7, 6), (8, 5), (9, 4), (10, 3), (11, 2), (12, 1), (13, 0),
        (6, 7), (5, 8), (4, 9), (3, 10), (2, 11), (1, 12), (0, 13)
    ]
    
    # Also try some non-standard: one point at center
    partitions.append((12, 1))  # already there
    
    for m_outer, k_inner in partitions:
        if time.time() > deadline - safety_margin:
            break
        
        if k_inner == 0:
            # Just a regular m-gon
            points = _generate_points(m_outer, 0, 0.0, 0.0)
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points
            continue
        
        # Multi-start Nelder-Mead optimization
        # Initial guesses for (r_ratio, rot_angle)
        initial_guesses = [
            (0.5, 0.0),
            (0.5, np.pi / k_inner),
            (0.3, 0.0),
            (0.7, 0.0),
            (0.5, np.pi / (2 * k_inner)),
            (0.4, np.pi / k_inner),
            (0.6, np.pi / k_inner),
            (0.5, 2 * np.pi / k_inner),
        ]
        
        for x0 in initial_guesses:
            if time.time() > deadline - safety_margin:
                break
            
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(m_outer, k_inner),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                if result.success or result.fun < 0:
                    r_ratio, rot_angle = result.x
                    r_ratio = max(0.01, min(0.99, r_ratio))
                    points = _generate_points(m_outer, k_inner, r_ratio, rot_angle)
                    s = _score(points)
                    if s > best_score:
                        best_score = s
                        best_points = points
            except Exception:
                pass
    
    # If we didn't find anything, fall back to regular 13-gon
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
