"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Minimum area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Area of convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area of polygon formed by points in convex position
        # Simple fallback: use all points as polygon (may overestimate)
        return _polygon_area(points)


def _polygon_area(vertices):
    """Shoelace formula for polygon area."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
    return 0.5 * abs(area)


def _make_mixture(m, k, r, phase, scale=1.0):
    """
    Create a point set as mixture of regular m-gon (radius r) and regular k-gon (radius 1.0).
    m + k should equal N = 13.
    phase: rotation of inner polygon relative to outer.
    """
    points = []
    # Outer regular k-gon at radius 1.0
    for i in range(k):
        angle = 2 * np.pi * i / k
        points.append([np.cos(angle), np.sin(angle)])
    # Inner regular m-gon at radius r, rotated by phase
    for i in range(m):
        angle = 2 * np.pi * i / m + phase
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)


def _score(points):
    """Compute score: min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    min_tri = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_tri / hull_area


def _optimize_partition(m, k):
    """
    Optimize over r (inner radius) and phase for partition m+k=N.
    Returns best (score, points) for this partition.
    """
    if m + k != N:
        return 0.0, None
    
    def neg_score(params):
        r, phase = params
        if r <= 0 or r >= 1:
            return 0.0
        pts = _make_mixture(m, k, r, phase)
        return -_score(pts)
    
    best_score = 0.0
    best_pts = None
    
    # Try multiple initial guesses
    for r0 in [0.3, 0.5, 0.7]:
        for phase0 in [0.0, np.pi / max(m, 1), np.pi / max(k, 1), np.pi / 4]:
            res = minimize(neg_score, [r0, phase0], method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < 0:
                r_opt, phase_opt = res.x
                r_opt = np.clip(r_opt, 0.05, 0.95)
                pts = _make_mixture(m, k, r_opt, phase_opt)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts
    
    # Also try swapping: inner m-gon at radius r, outer k-gon at radius 1
    # Already done above. Let's also try the reverse: inner k-gon, outer m-gon
    # But that's just swapping m and k, which we'll handle by trying both orders
    
    return best_score, best_pts


def construct():
    """
    Construct 13 points as a weighted mixture of two regular polygons.
    Try all partitions m+k=13 and optimize radii/phases.
    """
    best_score = 0.0
    best_points = None
    
    partitions = [(1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]
    
    for m, k in partitions:
        sc, pts = _optimize_partition(m, k)
        if pts is not None and sc > best_score:
            best_score = sc
            best_points = pts
    
    if best_points is None:
        # Fallback: regular 13-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
