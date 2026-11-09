# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from itertools import combinations


def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0


def _compute_min_triangle_area(points):
    """Compute the minimum area among all triangles formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _compute_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _compute_score(points):
    """Compute the score: A_min / A_hull."""
    a_min = _compute_min_triangle_area(points)
    a_hull = _compute_hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min):
        return 0.0
    return a_min / a_hull


def _compute_all_triangle_areas(points):
    """Compute areas of all C(n,3) triangles."""
    n = len(points)
    areas = np.array([])
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                areas = np.append(areas, area)
                indices.append((i, j, k))
    return areas, indices


def _gradient_ascent_optimize(points, max_iterations=5000, initial_step=0.02):
    """
    Optimize points using gradient ascent on the minimum triangle area.
    Uses finite differences to compute gradients and momentum for stability.
    """
    n = len(points)
    points = points.copy().astype(float)
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Momentum
    velocity = np.zeros_like(points)
    momentum = 0.9
    
    step_size = initial_step
    no_improve_count = 0
    patience = 100
    
    for iteration in range(max_iterations):
        # Compute all triangle areas
        areas, tri_indices = _compute_all_triangle_areas(points)
        
        if len(areas) == 0:
            break
        
        min_area = np.min(areas)
        
        # Find the minimum-area triangle(s)
        min_mask = areas < min_area * 1.001
        min_tris = [tri_indices[i] for i in range(len(areas)) if min_mask[i]]
        
        if not min_tris:
            break
        
        # Compute gradient of min_area with respect to each point
        # Using finite differences on the minimum area
        grad = np.zeros_like(points)
        eps = 1e-6
        
        # For efficiency, only compute gradients for points in min-area triangles
        affected_points = set()
        for tri in min_tris:
            for idx in tri:
                affected_points.add(idx)
        
        for pt_idx in affected_points:
            for dim in range(2):
                # Forward difference
                points_plus = points.copy()
                points_plus[pt_idx, dim] += eps
                
                # Compute min area with perturbed point
                areas_plus, _ = _compute_all_triangle_areas(points_plus)
                min_area_plus = np.min(areas_plus)
                
                # Only consider if this point is still in a min-area triangle
                # Otherwise, the gradient is zero (we're not on the active constraint)
                grad[pt_idx, dim] = (min_area_plus - min_area) / eps
        
        # Gradient ascent step
        velocity = momentum * velocity + (1 - momentum) * grad
        points_new = points + step_size * velocity
        
        # Project back: ensure points stay in reasonable bounds
        points_new = np.clip(points_new, -5, 5)
        
        # Check convex hull validity
        try:
            hull_area = _compute_hull_area(points_new)
            if hull_area <= 0:
                # Reject this step
                step_size *= 0.9
                continue
        except Exception:
            step_size *= 0.9
            continue
        
        new_score = _compute_score(points_new)
        
        if new_score > best_score * (1 + 1e-8):
            points = points_new
            best_points = points.copy()
            best_score = new_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            step_size *= 0.95
        
        if no_improve_count >= patience:
            break
    
    return best_points, best_score


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points.
    
    Uses a multi-start gradient-based optimization:
    1. Initialize points on various convex configurations
    2. Apply gradient ascent on the minimum triangle area
    3. Return the best configuration found
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points1 = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(points1)
    
    # 2. Regular 13-gon on larger circle
    points2 = np.column_stack([2 * np.cos(angles), 2 * np.sin(angles)])
    initializations.append(points2)
    
    # 3. Slightly perturbed regular 13-gon
    rng = np.random.RandomState(42)
    perturbation = rng.randn(n, 2) * 0.1
    points3 = points1 + perturbation
    initializations.append(points3)
    
    # 4. Points on two concentric circles (7 outer, 6 inner)
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    outer_points = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_points = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    points4 = np.vstack([outer_points, inner_points])
    initializations.append(points4)
    
    # 5. Points on three concentric circles (5, 4, 4)
    a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p1 = np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
    p3 = np.column_stack([0.3 * np.cos(a3), 0.3 * np.sin(a3)])
    points5 = np.vstack([p1, p2, p3])
    initializations.append(points5)
    
    # Optimize each initialization
    best_points = None
    best_score = -1
    
    for init_points in initializations:
        # Normalize to unit area hull for fair comparison
        try:
            hull = ConvexHull(init_points)
            hull_area = hull.volume
            if hull_area > 0:
                init_points = init_points / np.sqrt(hull_area)
        except Exception:
            continue
        
        optimized_points, score = _gradient_ascent_optimize(init_points, max_iterations=3000, initial_step=0.03)
        
        if score > best_score:
            best_score = score
            best_points = optimized_points
    
    if best_points is None:
        # Fallback: regular 13-gon
        return points1
    
    return best_points


# EVOLVE-BLOCK-END
