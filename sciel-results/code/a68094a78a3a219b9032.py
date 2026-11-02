"""Optimized placement of 11 points in a unit-side equilateral triangle."""
import time
import numpy as np
from itertools import combinations

N = 11
# Vertices of the equilateral triangle as specified in the interface
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, v0, v1, v2, eps=1e-9):
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
    # Optimization: Precompute pair differences? For N=11, 165 triangles, it's fast enough.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def farthest_point_sampling(n, seed=42):
    """Generate n points using farthest point sampling within the triangle."""
    rng = np.random.default_rng(seed)
    points = []
    
    # Start with a random point inside the triangle
    while True:
        u, v = rng.random(), rng.random()
        if u + v <= 1:
            p = (1 - u) * V0 + u * (1 - v) * V1 + u * v * V2
            points.append(p)
            break
    
    for _ in range(1, n):
        # Generate many random candidates
        candidates = []
        for _ in range(5000):
            u, v = rng.random(), rng.random()
            if u + v <= 1:
                p = (1 - u) * V0 + u * (1 - v) * V1 + u * v * V2
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

def local_search(points, time_limit=10.0):
    """
    Perform a greedy local search to maximize the minimum triangle area.
    Moves points slightly in random directions if it improves the score and stays inside.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = min_triangle_area(current_points)
    step_size = 0.05
    rng = np.random.default_rng(123)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try to move each point
        for i in range(N):
            if time.time() - start_time >= time_limit:
                break
            
            for _ in range(50): # 50 attempts per point per iteration
                if time.time() - start_time >= time_limit:
                    break
                
                # Random direction
                angle = rng.uniform(0, 2 * np.pi)
                new_p = current_points[i] + step_size * np.array([np.cos(angle), np.sin(angle)])
                
                # Check if inside
                if point_in_triangle(new_p, V0, V1, V2):
                    # Test the move
                    test_points = current_points.copy()
                    test_points[i] = new_p
                    test_score = min_triangle_area(test_points)
                    
                    if test_score > current_score:
                        current_points = test_points
                        current_score = test_score
                        improved = True
                        # Reduce step size slightly on improvement
                        step_size *= 0.95
                        break # Move to next point
        
        if not improved:
            # If no improvement in a full pass, reduce step size
            step_size *= 0.8
            if step_size < 1e-4:
                break
                
    return current_points

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s/150s limit
    
    best_points = None
    best_score = -float('inf')
    
    # Generate multiple initial configurations using FPS with different seeds
    # and run local search on each
    n_restarts = 5
    
    for seed in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Time remaining for this restart
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Allocate time for local search
        # Leave some time for evaluation
        ls_time = max(2.0, time_remaining / 2.0)
        
        # 1. Initial Sampling
        initial_points = farthest_point_sampling(N, seed=seed * 100 + 42)
        
        # Evaluate initial
        if all(point_in_triangle(p, V0, V1, V2) for p in initial_points):
            score = min_triangle_area(initial_points)
            if score > best_score:
                best_score = score
                best_points = initial_points.copy()
        
        # 2. Local Search Refinement
        optimized_points = local_search(initial_points, time_limit=ls_time)
        
        # Evaluate optimized
        if all(point_in_triangle(p, V0, V1, V2) for p in optimized_points):
            score = min_triangle_area(optimized_points)
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                
    # Final safety check
    if best_points is None:
        # Fallback: Return a valid set of points (e.g., centroid + vertices scaled)
        # Just return the first FPS result
        best_points = farthest_point_sampling(N, seed=42)
        
    # Ensure final output is strictly inside
    # (FPS and Local Search already enforce this, but double check)
    for i in range(N):
        if not point_in_triangle(best_points[i], V0, V1, V2):
            # This should not happen, but if it does, project to centroid
            best_points[i] = (V0 + V1 + V2) / 3.0
            
    return best_points
