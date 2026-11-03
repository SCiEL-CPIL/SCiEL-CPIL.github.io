"""Rewritten program for Heilbronn 13-point problem using concentric circle parameterization."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute min triangle area / convex hull area."""
    n = len(points)
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        hull_area = 1.0
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area


def _compute_score_vectorized(points):
    """Vectorized score computation for speed."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        hull_area = 1.0
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Concentric circle initialization
    # 7 points on outer ring (hull), 6 points on inner rings
    n_outer = 7
    n_inner = 6
    
    # Outer ring: 7 points on a circle of radius 1
    theta_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer_points = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    
    # Inner points: distributed on concentric circles
    # Use golden angle spiral for inner points
    inner_points = []
    for i in range(n_inner):
        # Distribute on circles of decreasing radius
        r = 0.5 * (1 - i / n_inner) ** 0.5
        theta = i * 2.399963229728653  # Golden angle
        inner_points.append([r * np.cos(theta), r * np.sin(theta)])
    inner_points = np.array(inner_points)
    
    points = np.vstack([outer_points, inner_points])
    # Center the points
    points -= np.mean(points, axis=0)
    
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Precompute triangle indices for fast evaluation
    tri_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def fast_score(pts):
        """Fast score using precomputed indices."""
        min_area = float('inf')
        for i, j, k in tri_indices:
            ax, ay = pts[i]
            bx, by = pts[j]
            cx, cy = pts[k]
            area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
            if area < min_area:
                min_area = area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            hull_area = 1.0
        if hull_area < 1e-12:
            return 0.0
        return min_area / hull_area
    
    # Local optimization: perturb points in worst triangles
    max_iterations = 50000
    step_size = 0.05
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Find the worst (minimum area) triangle
        min_area = float('inf')
        worst_tri = None
        for i, j, k in tri_indices:
            ax, ay = best_points[i]
            bx, by = best_points[j]
            cx, cy = best_points[k]
            area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
            if area < min_area:
                min_area = area
                worst_tri = (i, j, k)
        
        if worst_tri is None:
            break
        
        # Perturb the three points in the worst triangle
        i, j, k = worst_tri
        improved = False
        
        # Try small random perturbations
        for _ in range(20):
            if time.time() > deadline:
                break
            
            trial_points = best_points.copy()
            for idx in [i, j, k]:
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                trial_points[idx] += [dx, dy]
            
            trial_score = fast_score(trial_points)
            if trial_score > best_score:
                best_points = trial_points
                best_score = trial_score
                improved = True
        
        if not improved:
            # Reduce step size occasionally
            step_size *= 0.999
            if step_size < 1e-6:
                break
    
    # Final evaluation
    final_score = _compute_score(best_points)
    if final_score > best_score:
        best_score = final_score
    
    return best_points

# EVOLVE-BLOCK-END
