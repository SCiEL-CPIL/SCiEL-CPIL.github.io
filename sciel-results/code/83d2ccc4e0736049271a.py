"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3:
        return 0.0
    
    # Find the leftmost point (lowest x, then lowest y)
    start_idx = np.lexsort((points[:, 1], points[:, 0]))[0]
    
    # Sort remaining points by polar angle around the start point
    def polar_angle(p):
        dx = p[0] - points[start_idx, 0]
        dy = p[1] - points[start_idx, 1]
        if dx == 0 and dy == 0:
            return 0.0
        return math.atan2(dy, dx)
    
    indices = [start_idx] + [i for i in range(n) if i != start_idx]
    indices.sort(key=lambda i: (polar_angle(points[i]), np.sum((points[i] - points[start_idx])**2)))
    
    # Andrew's monotone chain
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for i in indices:
        p = points[i]
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    
    upper = []
    for i in reversed(indices):
        p = points[i]
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    
    # Shoelace formula
    m = len(hull)
    if m < 3:
        return 0.0
    area = 0.0
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0

def _min_triangle_area(points):
    """Compute the minimum area triangle among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) 
                               - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _regular_heptagon_displaced(offsets):
    """
    Generate 14 points from a regular heptagon with radial displacements.
    offsets: array of 7 values, each in [-1, 1]
    Point 2*i is at angle 2*pi*i/7 with radius 1 + offsets[i]
    Point 2*i+1 is at angle 2*pi*i/7 with radius 1 - offsets[i]
    This creates two interleaved heptagons.
    """
    angles = 2 * np.pi * np.arange(7) / 7
    points = np.zeros((14, 2))
    for i in range(7):
        r_outer = 1.0 + offsets[i]
        r_inner = 1.0 - offsets[i]
        points[2*i] = r_outer * np.array([np.cos(angles[i]), np.sin(angles[i])])
        points[2*i+1] = r_inner * np.array([np.cos(angles[i]), np.sin(angles[i])])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with regular 14-gon (all offsets = 0)
    initial_offsets = np.zeros(7)
    points = _regular_heptagon_displaced(initial_offsets)
    best_score = _score(points)
    best_offsets = initial_offsets.copy()
    
    # Simple gradient-free optimization: coordinate descent with random restarts
    # Use a combination of local search and random perturbations
    
    rng = np.random.RandomState(N)
    
    # Local search function
    def local_search(offsets, max_iter=200):
        current = offsets.copy()
        current_score = _score(_regular_heptagon_displaced(current))
        step_size = 0.1
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(7):
                # Try increasing and decreasing offset i
                for direction in [1, -1]:
                    trial = current.copy()
                    trial[i] += direction * step_size
                    # Clamp to reasonable range
                    trial[i] = np.clip(trial[i], -0.5, 0.5)
                    trial_score = _score(_regular_heptagon_displaced(trial))
                    if trial_score > current_score:
                        current = trial
                        current_score = trial_score
                        improved = True
                        break
                if not improved and direction == 1:
                    # Try smaller step if no improvement
                    pass
            if not improved:
                step_size *= 0.8
                if step_size < 1e-6:
                    break
        return current, current_score
    
    # Run local search from initial point
    best_offsets, best_score = local_search(best_offsets)
    
    # Try random restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initial offsets
        rand_offsets = rng.uniform(-0.3, 0.3, 7)
        rand_offsets, rand_score = local_search(rand_offsets, max_iter=100)
        
        if rand_score > best_score:
            best_score = rand_score
            best_offsets = rand_offsets
    
    # Final polish: very fine local search
    best_offsets, best_score = local_search(best_offsets, max_iter=500)
    
    points = _regular_heptagon_displaced(best_offsets)
    return points
# EVOLVE-BLOCK-END
