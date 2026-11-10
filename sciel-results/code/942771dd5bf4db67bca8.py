# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay


def _triangle_area(p1, p2, p3):
    """Compute area of triangle given three points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0


def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns (areas, indices)."""
    n = len(pts)
    idx = []
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(pts[i], pts[j], pts[k])
                idx.append((i, j, k))
                areas.append(a)
    return np.array(areas), idx


def _score(pts):
    """Compute the Heilbronn score: A_min / A_hull."""
    n = len(pts)
    if n < 3:
        return 0.0
    areas, idx = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    a_min = np.min(areas)
    if a_min < 1e-15:
        return 0.0
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull < 1e-15:
        return 0.0
    return a_min / a_hull


def _delaunay_forces(pts):
    """
    Compute forces based on Delaunay triangulation.
    For each Delaunay triangle, if its area is small, push vertices apart.
    Also add repulsive forces between all point pairs to avoid clustering.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Delaunay-based forces
    try:
        tri = Delaunay(pts)
        for simplex in tri.simplices:
            i, j, k = simplex
            area = _triangle_area(pts[i], pts[j], pts[k])
            if area < 1e-15:
                continue
            
            # Centroid of the triangle
            centroid = (pts[i] + pts[j] + pts[k]) / 3.0
            
            # Push vertices away from centroid, stronger for smaller triangles
            # Normalize by typical area (use 1/13 of hull area as reference)
            strength = 1.0 / (area + 0.01)
            
            for v in [i, j, k]:
                diff = pts[v] - centroid
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    forces[v] += strength * diff / norm * 0.05
    except Exception:
        pass
    
    # Pairwise repulsive forces (Coulomb-like)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random small perturbation
                forces[i] += np.random.randn(2) * 0.01
                forces[j] -= np.random.randn(2) * 0.01
            else:
                # Repulsion strength inversely proportional to distance squared
                repulsion = 0.1 / (dist * dist + 0.01)
                forces[i] += repulsion * diff / dist
                forces[j] -= repulsion * diff / dist
    
    return forces


def _laplacian_smoothing(pts, weights=None):
    """
    Apply Laplacian smoothing to move points toward the average of their neighbors.
    This helps maintain a well-spread configuration.
    """
    n = len(pts)
    if weights is None:
        weights = np.ones((n, n))
    
    smoothed = np.zeros_like(pts)
    for i in range(n):
        total_weight = 0
        weighted_sum = np.zeros(2)
        for j in range(n):
            if i != j:
                w = weights[i, j]
                weighted_sum += w * pts[j]
                total_weight += w
        if total_weight > 1e-10:
            smoothed[i] = weighted_sum / total_weight
        else:
            smoothed[i] = pts[i]
    
    return smoothed


def _compute_weights(pts):
    """
    Compute distance-based weights for Laplacian smoothing.
    Closer points have higher weight.
    """
    n = len(pts)
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(pts[i] - pts[j])
                weights[i, j] = 1.0 / (dist + 0.1)
    return weights


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area, using Delaunay-based physics relaxation
    with Laplacian smoothing.
    """
    deadline = time.time() + 600.0
    n = 13
    
    # Start with a regular 13-gon as initial configuration
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_pts)
    
    # Multiple initial configurations
    initial_configs = [best_pts.copy()]
    
    # Configuration 2: 12 points on circle + 1 center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts2 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    pts2 = np.vstack([pts2, np.array([[0.0, 0.0]])])
    initial_configs.append(pts2)
    
    # Configuration 3: Two concentric rings (7 + 6)
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    pts3 = np.vstack([
        np.column_stack([np.cos(angles7) * 1.0, np.sin(angles7) * 1.0]),
        np.column_stack([np.cos(angles6) * 0.5, np.sin(angles6) * 0.5])
    ])
    initial_configs.append(pts3)
    
    # Configuration 4: Three concentric rings (5 + 4 + 4)
    angles5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    angles4a = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    angles4b = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    pts4 = np.vstack([
        np.column_stack([np.cos(angles5) * 1.0, np.sin(angles5) * 1.0]),
        np.column_stack([np.cos(angles4a) * 0.6, np.sin(angles4a) * 0.6]),
        np.column_stack([np.cos(angles4b) * 0.3, np.sin(angles4b) * 0.3])
    ])
    initial_configs.append(pts4)
    
    # Configuration 5: Grid-like pattern
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if len(grid_pts) < 13:
                grid_pts.append([i * 0.8 - 1.2, j * 0.8 - 1.2])
    grid_pts = np.array(grid_pts[:13])
    initial_configs.append(grid_pts)
    
    # Configuration 6: Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = []
    for i in range(n):
        r = np.sqrt(i + 1) * 0.3
        theta = i * golden_angle
        fib_pts.append([r * np.cos(theta), r * np.sin(theta)])
    fib_pts = np.array(fib_pts)
    initial_configs.append(fib_pts)
    
    for init_pts in initial_configs:
        if time.time() > deadline - 60:
            break
        
        pts = init_pts.copy()
        current_score = _score(pts)
        
        # Main optimization loop
        step_size = 0.03
        cooling = 1.0
        smooth_alpha = 0.1
        
        max_iters = 8000
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            # Compute forces
            forces = _delaunay_forces(pts)
            
            # Apply Laplacian smoothing to maintain spread
            weights = _compute_weights(pts)
            smoothed = _laplacian_smoothing(pts, weights)
            
            # Blend: move toward smoothed position + force-driven update
            pts_new = pts + step_size * cooling * forces + smooth_alpha * (smoothed - pts)
            
            # Bound check
            max_norm = np.max(np.linalg.norm(pts_new, axis=1))
            if max_norm > 5:
                pts_new = pts_new / max_norm * 5
            
            # Check for degeneracy
            if np.any(np.isfinite(pts_new) == False):
                pts_new = pts.copy()
            
            new_score = _score(pts_new)
            
            if new_score > current_score:
                current_score = new_score
                pts = pts_new
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            else:
                # Revert and cool down
                cooling *= 0.97
                step_size *= 0.995
                smooth_alpha *= 0.99
            
            if cooling < 0.001 or step_size < 0.001:
                break
        
        # Final refinement with smaller steps
        pts = best_pts.copy()
        current_score = _score(pts)
        step_size = 0.005
        cooling = 1.0
        smooth_alpha = 0.05
        
        for it in range(5000):
            if time.time() > deadline - 15:
                break
            
            forces = _delaunay_forces(pts)
            weights = _compute_weights(pts)
            smoothed = _laplacian_smoothing(pts, weights)
            
            pts_new = pts + step_size * cooling * forces + smooth_alpha * (smoothed - pts)
            
            max_norm = np.max(np.linalg.norm(pts_new, axis=1))
            if max_norm > 5:
                pts_new = pts_new / max_norm * 5
            
            if np.any(np.isfinite(pts_new) == False):
                pts_new = pts.copy()
            
            new_score = _score(pts_new)
            
            if new_score > current_score:
                current_score = new_score
                pts = pts_new
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            else:
                cooling *= 0.98
                step_size *= 0.995
                smooth_alpha *= 0.995
            
            if cooling < 0.001 or step_size < 0.0005:
                break
    
    # Ensure valid output
    if not np.all(np.isfinite(best_pts)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Final check: ensure convex hull exists
    try:
        hull = ConvexHull(best_pts)
    except Exception:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts


# EVOLVE-BLOCK-END
