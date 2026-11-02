"""Optimized placement of 11 points in a unit-area equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
# Vertices of the equilateral triangle with area 1
# Area = (1/2) * base * height = 1 => base * height = 2
# For equilateral: height = (sqrt(3)/2) * base => base^2 * sqrt(3)/2 = 2 => base^2 = 4/sqrt(3)
# base = 2 / 3^(1/4), height = 2 * 3^(1/4) / 2 = 3^(1/4) * 2 / 2? Let's recompute.
# Let side length be s. Area = (sqrt(3)/4) * s^2 = 1 => s^2 = 4/sqrt(3) => s = 2 * 3^(-1/4)
# height = s * sqrt(3)/2 = 2 * 3^(-1/4) * sqrt(3)/2 = 3^(1/2) * 3^(-1/4) = 3^(1/4)
# So vertices: (0,0), (s, 0), (s/2, h) where s = 2*3**(-0.25), h = 3**0.25
S = 2 * 3 ** (-0.25)
H = 3 ** 0.25
V0 = np.array([0.0, 0.0])
V1 = np.array([S, 0.0])
V2 = np.array([S / 2, H])

def point_in_triangle(p, v0, v1, v2, eps=1e-10):
    """Check if point p is inside or on the boundary of triangle v0, v1, v2."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross product|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective_flat(x):
    """Objective function for flat array: minimize negative min area."""
    points = x.reshape((N, 2))
    # Check if all points are inside the triangle
    for i in range(N):
        if not point_in_triangle(points[i], V0, V1, V2):
            return 1e10  # Large penalty
    area = min_triangle_area(points)
    return -area

def farthest_point_sampling(n, v0, v1, v2, seed=42):
    """Generate n points using farthest point sampling within the triangle."""
    rng = np.random.default_rng(seed)
    points = []
    
    # Start with a random point inside the triangle
    while True:
        u, v = rng.random(), rng.random()
        if u + v <= 1:
            p = (1 - u) * v0 + u * (1 - v) * v1 + u * v * v2
            points.append(p)
            break
    
    for _ in range(1, n):
        # Generate many random candidates
        candidates = []
        for _ in range(2000):
            u, v = rng.random(), rng.random()
            if u + v <= 1:
                p = (1 - u) * v0 + u * (1 - v) * v1 + u * v * v2
                candidates.append(p)
        
        candidates = np.array(candidates)
        
        # Find the candidate that maximizes the minimum distance to existing points
        best_idx = 0
        best_min_dist = -1
        for i, c in enumerate(candidates):
            dists = np.linalg.norm(points - c, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        
        points.append(candidates[best_idx])
    
    return np.array(points)

def project_to_triangle(p):
    """Project point p onto the triangle if it's outside."""
    if point_in_triangle(p, V0, V1, V2):
        return p
    
    # Simple projection: move towards the centroid
    centroid = (V0 + V1 + V2) / 3
    for t in np.linspace(0, 1, 100):
        proj = p + t * (centroid - p)
        if point_in_triangle(proj, V0, V1, V2):
            return proj
    return centroid

def optimize_single(points, max_iter=100, time_limit=10):
    """Optimize a single configuration using Powell's method."""
    start_time = time.time()
    x0 = points.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        points = x.reshape((N, 2))
        # Project points to triangle if needed
        for i in range(N):
            points[i] = project_to_triangle(points[i])
        return objective_flat(points.flatten())
    
    try:
        result = minimize(objective, x0, method='Powell', options={'maxiter': max_iter, 'maxfev': 5000})
        if result.success or result.fun < objective_flat(x0):
            return result.x.reshape((N, 2))
    except TimeoutError:
        pass
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_points = None
    best_score = -float('inf')
    
    # Generate initial configurations using FPS with different seeds
    n_initial = 5
    initial_configs = []
    for seed in range(n_initial):
        if time.time() > deadline:
            break
        points = farthest_point_sampling(N, V0, V1, V2, seed=seed * 100 + 42)
        initial_configs.append(points)
    
    # Also add a grid-based initial configuration
    # Divide triangle into smaller triangles and pick centroids
    grid_points = []
    for i in range(1, 4):
        for j in range(1, 4 - i + 1):
            # Barycentric coordinates
            u = i / 4
            v = j / 4
            w = 1 - u - v
            if w >= -1e-10:
                p = u * V0 + v * V1 + w * V2
                grid_points.append(p)
    if len(grid_points) >= N:
        initial_configs.append(np.array(grid_points[:N]))
    
    # Optimize each initial configuration
    for i, points in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Evaluate initial score
        if all(point_in_triangle(p, V0, V1, V2) for p in points):
            score = min_triangle_area(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        # Optimize
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        max_iter = max(10, int(100 * time_remaining / 30))
        time_limit = max(2, time_remaining / 2)
        
        optimized = optimize_single(points, max_iter=max_iter, time_limit=time_limit)
        
        if all(point_in_triangle(p, V0, V1, V2) for p in optimized):
            score = min_triangle_area(optimized)
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
    
    # If we have time, do a few more random restarts
    if best_points is None:
        # Fallback: return FPS points
        best_points = farthest_point_sampling(N, V0, V1, V2, seed=42)
    else:
        # Try a few more random perturbations of the best solution
        n_extra = 3
        for _ in range(n_extra):
            if time.time() > deadline:
                break
            
            perturbed = best_points + np.random.randn(N, 2) * 0.05
            perturbed = np.array([project_to_triangle(p) for p in perturbed])
            
            if all(point_in_triangle(p, V0, V1, V2) for p in perturbed):
                score = min_triangle_area(perturbed)
                if score > best_score:
                    best_score = score
                    best_points = perturbed.copy()
                
                time_remaining = deadline - time.time()
                if time_remaining > 5:
                    optimized = optimize_single(perturbed, max_iter=50, time_limit=time_remaining / 2)
                    if all(point_in_triangle(p, V0, V1, V2) for p in optimized):
                        score = min_triangle_area(optimized)
                        if score > best_score:
                            best_score = score
                            best_points = optimized.copy()
    
    return best_points
