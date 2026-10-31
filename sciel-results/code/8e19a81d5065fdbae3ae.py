"""Optimized program for Heilbronn convex regions - 13 points maximizing min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time
import itertools

N = 13

def compute_min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def compute_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score_function(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = compute_min_triangle_area(points)
    return min_area / hull_area

def generate_hexagonal_pattern():
    """Hexagonal-like arrangement with center + 6 inner + 6 outer."""
    points = [[0, 0]]
    for i in range(6):
        angle = i * np.pi / 3
        points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    return np.array(points)

def generate_circular_pattern():
    """13 points on a circle."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def generate_random_pattern(seed):
    """Random points in unit square."""
    rng = np.random.RandomState(seed)
    return rng.rand(N, 2)

def delaunay_edge_optimization(points, deadline, max_iter=100):
    """
    Optimize points by focusing on Delaunay triangle areas.
    The key insight: at the optimum, the minimum-area triangle is likely a Delaunay triangle.
    We use the Delaunay structure to guide optimization.
    """
    start_time = time.time()
    current = points.copy()
    current_score = score_function(current)
    best_points = current.copy()
    best_score = current_score
    
    # Get Delaunay triangles
    try:
        tri = Delaunay(current)
        simplices = tri.simplices
    except Exception:
        return best_points, best_score
    
    # Objective: maximize min area over Delaunay triangles (proxy for global min)
    def delaunay_min_area(params):
        pts = params.reshape(N, 2)
        # Compute areas of all Delaunay triangles
        min_area = float('inf')
        for tri_idx in simplices:
            p0, p1, p2 = pts[tri_idx[0]], pts[tri_idx[1]], pts[tri_idx[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
            if area < min_area:
                min_area = area
        return min_area
    
    def neg_delaunay_min(params):
        return -delaunay_min_area(params)
    
    # Optimize using Nelder-Mead on the full parameter set,
    # but with the Delaunay-aware objective to guide the search
    try:
        result = minimize(
            neg_delaunay_min,
            current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9}
        )
        
        if time.time() > deadline:
            return best_points, best_score
        
        new_points = result.x.reshape(N, 2)
        new_score = score_function(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            
    except Exception:
        pass
    
    return best_points, best_score

def multi_stage_optimize(start_points, deadline):
    """
    Multi-stage optimization:
    1. Coarse Nelder-Mead on all 26 params
    2. Delaunay-guided refinement
    3. Fine-tuning with smaller steps
    """
    current = start_points.copy()
    current_score = score_function(current)
    best_points = current.copy()
    best_score = current_score
    
    # Stage 1: Coarse optimization
    def neg_score(params):
        pts = params.reshape(N, 2)
        return -score_function(pts)
    
    try:
        result = minimize(
            neg_score,
            current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 80, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if time.time() > deadline:
            return best_points, best_score
        
        new_points = result.x.reshape(N, 2)
        new_score = score_function(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            current = new_points
            
    except Exception:
        pass
    
    # Stage 2: Delaunay-guided optimization
    if time.time() < deadline - 10:
        current, score = delaunay_edge_optimization(current, deadline, max_iter=80)
        if score > best_score:
            best_points = current
            best_score = score
    
    # Stage 3: Fine-tuning
    if time.time() < deadline - 5:
        try:
            result = minimize(
                neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 60, 'xatol': 1e-8, 'fatol': 1e-10, 'initial_simplex': 1e-4}
            )
            
            if time.time() > deadline:
                return best_points, best_score
            
            new_points = result.x.reshape(N, 2)
            new_score = score_function(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
                
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # Generate diverse starting configurations
    start_configs = [
        generate_hexagonal_pattern(),
        generate_circular_pattern(),
        generate_random_pattern(42),
        generate_random_pattern(123),
        generate_random_pattern(999),
        generate_random_pattern(777),
    ]
    
    best_points = start_configs[0]
    best_score = score_function(best_points)
    
    # Try multi-stage optimization from each starting configuration
    for i, start in enumerate(start_configs):
        if time.time() > deadline:
            break
        
        # Normalize starting configuration
        start_scaled = start.copy()
        center = start_scaled.mean(axis=0)
        start_scaled = start_scaled - center
        
        # Ensure reasonable scale
        norm = np.linalg.norm(start_scaled, axis=1).mean()
        if norm > 0:
            start_scaled = start_scaled * (1.0 / norm)
        
        optimized, score = multi_stage_optimize(start_scaled, deadline)
        
        if score > best_score:
            best_score = score
            best_points = optimized
    
    # Final verification
    final_score = score_function(best_points)
    if final_score > best_score:
        best_score = final_score
    
    return best_points
