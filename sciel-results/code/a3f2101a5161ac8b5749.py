# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import itertools
from scipy.optimize import minimize

N = 13
# Precompute all 286 triangle index triples
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """Hard Heilbronn score: A_min / A_hull."""
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 1e-12:
        return 0.0
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas) / A_hull

def compute_all_areas(pts):
    """Vectorized areas of all 286 triangles."""
    p1 = pts[TRI_IDX[:, 0]]
    p2 = pts[TRI_IDX[:, 1]]
    p3 = pts[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.abs(cross) * 0.5

def delaunay_repair(points, rng, max_iters=50, time_limit=None):
    """
    Delaunay-based repair: identify small-area triangles in the Delaunay triangulation
    and move their vertices to increase area.
    """
    pts = points.copy()
    current_score = compute_score(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    start = time.time()
    for iteration in range(max_iters):
        if time_limit and time.time() > time_limit:
            break
        
        try:
            delaunay = Delaunay(pts)
        except Exception:
            break
        
        # Get Delaunay triangles
        tri_indices = delaunay.simplices  # shape (n_tris, 3)
        
        # Compute areas of Delaunay triangles
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = np.abs(cross) * 0.5
        
        # Find the smallest triangle
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Get the three vertices of the smallest triangle
        tri = tri_indices[min_idx]
        
        # For each vertex in this triangle, try moving it
        improved = False
        for v_idx in tri:
            # Try several directions
            for _ in range(5):
                theta = rng.uniform(0, 2 * np.pi)
                mag = 0.02 * rng.uniform(0.5, 1.5)
                dx = mag * np.cos(theta)
                dy = mag * np.sin(theta)
                
                old_x, old_y = pts[v_idx, 0], pts[v_idx, 1]
                pts[v_idx, 0] += dx
                pts[v_idx, 1] += dy
                
                new_score = compute_score(pts)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
                else:
                    pts[v_idx, 0] = old_x
                    pts[v_idx, 1] = old_y
        
        if not improved:
            break
    
    return best_pts, best_score

def gradient_ascent(points, rng, time_limit, step_size_init=0.05):
    """
    Gradient ascent using random perturbations, accepting only improvements.
    """
    pts = points.copy()
    current_score = compute_score(pts)
    best_score = current_score
    best_pts = pts.copy()
    step_size = step_size_init
    
    start = time.time()
    while time.time() < time_limit and step_size > 1e-6:
        improved = False
        for idx in rng.permutation(N):
            for _ in range(2):
                theta = rng.uniform(0, 2 * np.pi)
                mag = step_size * rng.uniform(0.3, 1.0)
                dx = mag * np.cos(theta)
                dy = mag * np.sin(theta)
                
                old_x, old_y = pts[idx, 0], pts[idx, 1]
                pts[idx, 0] += dx
                pts[idx, 1] += dy
                
                new_score = compute_score(pts)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
                else:
                    pts[idx, 0] = old_x
                    pts[idx, 1] = old_y
        
        if not improved:
            step_size *= 0.6
        else:
            step_size *= 0.9
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -1.0
    
    candidates = []
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 7 outer, 6 inner
    ang7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p7 = np.column_stack([np.cos(ang7), np.sin(ang7)])
    ang6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    p6 = np.column_stack([0.55 * np.cos(ang6), 0.55 * np.sin(ang6)])
    candidates.append(np.vstack([p7, p6]))
    
    # 3. Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    fib_angles = 2 * np.pi * np.arange(N) / (golden ** 2)
    fib_radii = np.sqrt(np.arange(N) / N)
    candidates.append(np.column_stack([fib_radii * np.cos(fib_angles),
                                       fib_radii * np.sin(fib_angles)]))
    
    # 4. Random
    for _ in range(5):
        candidates.append(rng.random((N, 2)) * 2 - 1)
    
    # 5. Perturbed structured
    for base_idx in [0, 1, 2]:
        if base_idx < len(candidates):
            candidates.append(candidates[base_idx] + rng.normal(0, 0.03, size=(N, 2)))
            
    n_cand = len(candidates)
    
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline - 5:
            break
            
        init_score = compute_score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()
            
        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        remaining_cands = n_cand - i
        time_per_cand = min(remaining / max(remaining_cands, 1), 60.0)
        cand_deadline = time.time() + time_per_cand
        
        cur = init_pts.copy()
        
        # Phase 1: Delaunay repair (structural, targets small triangles directly)
        if time.time() < cand_deadline - 20:
            dr_pts, dr_score = delaunay_repair(cur, rng, max_iters=30, time_limit=cand_deadline - 20)
            if dr_score > best_score:
                best_score = dr_score
                best_points = dr_pts.copy()
            cur = dr_pts
        
        # Phase 2: Gradient ascent for fine-tuning
        if time.time() < cand_deadline - 5:
            ga_pts, ga_score = gradient_ascent(cur, rng, cand_deadline - 5, step_size_init=0.03)
            if ga_score > best_score:
                best_score = ga_score
                best_points = ga_pts.copy()
            cur = ga_pts
            
    # Final Polish: Delaunay repair + gradient ascent
    if best_points is not None and time.time() < deadline - 10:
        # Delaunay repair
        dr_pts, dr_score = delaunay_repair(best_points, rng, max_iters=50, time_limit=deadline - 20)
        if dr_score > best_score:
            best_score = dr_score
            best_points = dr_pts.copy()
        
        # Gradient ascent
        if time.time() < deadline - 5:
            ga_pts, ga_score = gradient_ascent(best_points, rng, deadline - 5, step_size_init=0.01)
            if ga_score > best_score:
                best_score = ga_score
                best_points = ga_pts.copy()
            
    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
