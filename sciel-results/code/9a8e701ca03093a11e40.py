# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize
import random

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _get_triangle_areas(points):
    """
    Efficiently compute all triangle areas and return the array.
    """
    n = points.shape[0]
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _local_nudge(points, max_iters=2000, step=0.01):
    """
    Heuristic local search: 
    1. Find the smallest triangle.
    2. Calculate gradients of its area w.r.t its vertices.
    3. Nudge vertices away from each other to increase area.
    4. If score improves, accept; else reduce step size.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    if current_score == 0:
        return current_points, current_score

    step_size = step
    no_improve_count = 0
    
    for i in range(max_iters):
        # Get areas
        areas = _get_triangle_areas(current_points)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Reconstruct indices
        # This is slightly inefficient but max_iters is low
        idx = np.arange(13)
        triples = list(itertools.combinations(idx, 3))
        i1, i2, i3 = triples[min_idx]
        
        p1 = current_points[i1]
        p2 = current_points[i2]
        p3 = current_points[i3]
        
        # Gradient of area w.r.t p1, p2, p3
        # Area = 0.5 * | (p2-p1) x (p3-p1) |
        # To maximize area, we want to move points to increase the cross product magnitude.
        # A simpler heuristic: move each point away from the centroid of the other two?
        # Or move p1 in direction of -grad, p2 in direction of -grad, etc.
        
        # Let's use a simple repulsion force based on the gradient of the squared area (ignoring sign for a moment, but we want to increase area)
        # Actually, if the triangle is small, we just want to push the vertices apart.
        
        # Direction for p1: away from the line p2-p3?
        # Vector p2-p3. Normal to it.
        # Let's just try moving each point slightly away from the centroid of the triangle.
        centroid = (p1 + p2 + p3) / 3.0
        
        d1 = p1 - centroid
        d2 = p2 - centroid
        d3 = p3 - centroid
        
        # Normalize directions
        norm1 = np.linalg.norm(d1)
        norm2 = np.linalg.norm(d2)
        norm3 = np.linalg.norm(d3)
        
        if norm1 > 1e-6: d1 = d1 / norm1
        else: d1 = np.array([1.0, 0.0])
        
        if norm2 > 1e-6: d2 = d2 / norm2
        else: d2 = np.array([0.0, 1.0])
        
        if norm3 > 1e-6: d3 = d3 / norm3
        else: d3 = np.array([-1.0, 0.0])
        
        # Apply nudge
        new_points = current_points.copy()
        new_points[i1] += step_size * d1
        new_points[i2] += step_size * d2
        new_points[i3] += step_size * d3
        
        # Keep points within bounds [0.01, 0.99]
        new_points = np.clip(new_points, 0.01, 0.99)
        
        new_score = _compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                step_size *= 0.5
                if step_size < 1e-5:
                    break
                no_improve_count = 0
                
    return current_points, current_score

def _generate_architectural_templates():
    """
    Generate diverse, symmetric initial configurations.
    """
    templates = []
    
    # 1. Hexagonal Lattice Subset (Center + 6 neighbors + 6 outer)
    # This is the most promising structure for Heilbronn.
    # Triangular lattice vectors:
    # u = (1, 0), v = (0.5, sqrt(3)/2)
    # Points:
    # (0,0)
    # (1,0), (-1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (0.5, -sqrt(3)/2), (-0.5, -sqrt(3)/2)
    # (2,0), (1.5, sqrt(3)/2), (0.5, sqrt(3)), (-0.5, sqrt(3)), (-1.5, sqrt(3)/2), (-2,0) ... 
    # Let's build a specific 13-point cluster.
    
    # Standard hexagonal packing of 13:
    # Center (0,0)
    # Ring 1: 6 points at distance 1
    # Ring 2: 6 points at distance 2 (subset of 12)
    
    pts = [[0, 0]]
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    
    # Ring 2: Pick 6 points from the 12 available to maintain symmetry and avoid crowding
    # The 12 points are at angles k*pi/6 with radius 2? No, in triangular lattice, distance 2 neighbors are at angles 0, 60, 120... 
    # Actually, the second shell of a triangular lattice has 12 points.
    # To get 13 total, we take Center + 6 (Ring 1) + 6 (Ring 2).
    # We should pick Ring 2 points that are furthest from Ring 1 points? 
    # Or just align with the lattice.
    # Lattice points at distance 2:
    # (2, 0), (1, sqrt(3)), (-1, sqrt(3)), (-2, 0), (-1, -sqrt(3)), (1, -sqrt(3)) -> these are distance 2.
    # (1.5, 0.5*sqrt(3)) is distance sqrt(2.25+0.75)=sqrt(3) ~ 1.732. This is in Ring 1.5? No.
    # Let's just use the coordinates of the first 13 points of a triangular lattice sorted by distance from center.
    
    lattice_pts = []
    # Generate triangular lattice points in a box
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            lattice_pts.append([x, y])
            
    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the 13 closest
    hex_pts = lattice_pts[sorted_indices[:13]]
    # Center them
    hex_pts = hex_pts - np.mean(hex_pts, axis=0)
    # Scale to fit in [0,1]
    max_norm = np.max(np.linalg.norm(hex_pts, axis=1))
    if max_norm > 0:
        hex_pts = hex_pts / max_norm * 0.4 + 0.5
    templates.append(hex_pts)
    
    # 2. Concentric Rings (1 + 6 + 6)
    # Center
    pts = [[0.5, 0.5]]
    # Ring 1: 6 points
    r1 = 0.25
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([0.5 + r1 * np.cos(angle), 0.5 + r1 * np.sin(angle)])
    # Ring 2: 6 points, rotated by 30 degrees to avoid alignment
    r2 = 0.45
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts.append([0.5 + r2 * np.cos(angle), 0.5 + r2 * np.sin(angle)])
    templates.append(np.array(pts))
    
    # 3. Concentric Rings (1 + 5 + 7) - Alternative symmetry
    pts = [[0.5, 0.5]]
    r1 = 0.2
    for i in range(5):
        angle = i * 2 * np.pi / 5
        pts.append([0.5 + r1 * np.cos(angle), 0.5 + r1 * np.sin(angle)])
    r2 = 0.4
    for i in range(7):
        angle = i * 2 * np.pi / 7 + np.pi / 7
        pts.append([0.5 + r2 * np.cos(angle), 0.5 + r2 * np.sin(angle)])
    templates.append(np.array(pts))
    
    # 4. "Punched" 4x4 Grid
    # 4x4 grid = 16 points. Remove 3 corners or edges to get 13.
    # Removing corners usually helps increase min area if they are too close to edges.
    # Let's try removing (0,0), (3,0), (0,3)
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    # Remove indices 0, 3, 12
    mask = np.ones(16, dtype=bool)
    mask[0] = False
    mask[3] = False
    mask[12] = False
    punched_grid = grid[mask]
    punched_grid = punched_grid / 3.0 * 0.8 + 0.1
    templates.append(punched_grid)
    
    # 5. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    templates.append(pts)
    
    # 6. Randomized Hexagonal (perturbed lattice)
    rng = np.random.default_rng(123)
    hex_perturbed = hex_pts + rng.normal(0, 0.05, hex_pts.shape)
    hex_perturbed = np.clip(hex_perturbed, 0.01, 0.99)
    templates.append(hex_perturbed)
    
    # 7. Another lattice subset: Rectangular lattice 3x5 minus 2
    # 3x5 = 15. Remove 2.
    grid2 = np.array([[i, j] for i in range(3) for j in range(5)], dtype=float)
    mask2 = np.ones(15, dtype=bool)
    mask2[0] = False
    mask2[14] = False
    rect_grid = grid2[mask2]
    rect_grid = rect_grid / np.max(rect_grid) * 0.8 + 0.1
    templates.append(rect_grid)

    return templates

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using Architectural Template Search + Local Nudging.
    """
    start_time = time.time()
    deadline = 630  # seconds, strict limit
    safety_margin = 10
    effective_deadline = deadline - safety_margin
    
    templates = _generate_architectural_templates()
    
    best_points = None
    best_score = -np.inf
    
    # Iterate through templates
    for i, template in enumerate(templates):
        if time.time() - start_time > effective_deadline:
            break
            
        current_points = template.copy()
        current_score = _compute_score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Check time before expensive local search
        if time.time() - start_time > effective_deadline:
            break
            
        # Perform local nudging
        # Allocate time based on remaining budget
        time_left = effective_deadline - (time.time() - start_time)
        # Estimate: _compute_score is fast (~1ms). _local_nudge does ~2000 iters * 1ms = 2s.
        # We can afford ~100 local nudges in 600s if they are fast, but they might be slower.
        # Let's limit max_iters in _local_nudge based on time.
        # Actually, _local_nudge is fixed 2000 iters. Let's just run it if we have > 5s left.
        
        if time_left > 5:
            refined_points, refined_score = _local_nudge(current_points, max_iters=1000, step=0.02)
            
            if refined_score > current_score:
                current_points = refined_points
                current_score = refined_score
                
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
        # Optional: Second pass with smaller step if time permits
        if time.time() - start_time < effective_deadline - 20:
            refined_points2, refined_score2 = _local_nudge(current_points, max_iters=500, step=0.005)
            if refined_score2 > current_score:
                current_points = refined_points2
                current_score = refined_score2
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()

    # Fallback if all failed
    if best_points is None:
        best_points = templates[0]
        
    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = np.array([
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.1, 0.53], [0.36, 0.53], [0.63, 0.53], [0.9, 0.53],
            [0.1, 0.26], [0.36, 0.26], [0.63, 0.26], [0.9, 0.26],
            [0.3, 0.0], [0.7, 0.0]
        ])
        
    return best_points

# EVOLVE-BLOCK-END
