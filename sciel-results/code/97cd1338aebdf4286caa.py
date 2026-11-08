"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Find the point with the lowest y (and lowest x if tie)
    start = 0
    for i in range(1, n):
        if points[i][1] < points[start][1] or (points[i][1] == points[start][1] and points[i][0] < points[start][0]):
            start = i
    
    # Sort by polar angle from the start point
    def polar_angle(p):
        return np.arctan2(p[1] - points[start][1], p[0] - points[start][0])
    
    sorted_points = sorted(points, key=polar_angle)
    
    # Build the convex hull using the monotone chain algorithm (Andrew's algorithm)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Sort by x, then y
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    
    # Build lower hull
    lower = []
    for p in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in sorted_points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Concatenate to form the convex hull
    hull = lower[:-1] + upper[:-1]
    
    # Compute area using the shoelace formula
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    
    return abs(area) / 2.0

def _score(points):
    """Compute the score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _regular_ngon(n, radius=1.0, center=(0.0, 0.0), rotation=0.0):
    """Generate vertices of a regular n-gon."""
    angles = rotation + 2 * np.pi * np.arange(n) / n
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _two_ring_points(r_inner, r_outer, angle_offset, n_inner, n_outer):
    """Generate points on two concentric regular polygons."""
    inner = _regular_ngon(n_inner, r_inner, (0, 0), 0.0)
    outer = _regular_ngon(n_outer, r_outer, (0, 0), angle_offset)
    return np.vstack([inner, outer])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10-second safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initialize with regular 13-gon
    initial = _regular_ngon(N, 1.0, (0, 0), 0.0)
    initial_score = _score(initial)
    if initial_score > best_score:
        best_score = initial_score
        best_points = initial.copy()
    
    # Try two-ring configurations: inner m-gon and outer k-gon with m+k=13
    # Possible splits: (1,12), (2,11), (3,10), (4,9), (5,8), (6,7)
    splits = [(1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]
    
    for n_inner, n_outer in splits:
        # Try various radius ratios and angle offsets
        for r_inner in np.linspace(0.0, 0.95, 20):
            if time.time() > deadline:
                break
            r_outer = 1.0
            for angle_offset in np.linspace(0, 2 * np.pi, 15):
                points = _two_ring_points(r_inner, r_outer, angle_offset, n_inner, n_outer)
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        
        if time.time() > deadline:
            break
    
    # Local optimization using a simple gradient-free approach
    if best_points is not None and time.time() < deadline:
        # Use a coordinate descent / hill climbing approach
        current_points = best_points.copy()
        current_score = best_score
        step_size = 0.01
        no_improve_count = 0
        
        while time.time() < deadline - 5.0 and no_improve_count < 50:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                for coord in range(2):
                    for direction in [-1, 1]:
                        trial = current_points.copy()
                        trial[i, coord] += direction * step_size
                        trial_score = _score(trial)
                        if trial_score > current_score + 1e-10:
                            current_points = trial
                            current_score = trial_score
                            improved = True
                            no_improve_count = 0
                            break
                    if time.time() > deadline - 5.0:
                        break
            if not improved:
                no_improve_count += 1
                step_size *= 0.9
            if step_size < 1e-6:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
