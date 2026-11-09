import numpy as np
import time
from scipy.spatial import ConvexHull

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _subgradient(points):
    """
    Compute the subgradient of the minimum triangle area with respect to the points.
    Returns a gradient array of shape (13, 2) pointing in the direction that increases
    the minimum triangle area.
    """
    n = len(points)
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Signed double area: 2*signed_area = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
    signed_2area = (
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    areas = 0.5 * np.abs(signed_2area)
    
    # Find the minimum area and all triangles near the minimum
    a_min = np.min(areas)
    # Use a tolerance to include all "active" triangles
    tol = 1e-8 * a_min + 1e-12
    active = np.where(areas <= a_min + tol)[0]
    
    if len(active) == 0:
        return np.zeros((n, 2))
    
    # For each active triangle, compute the gradient of the area w.r.t. each of its 3 points
    # area = 0.5 * |signed_2area|
    # d(area)/d(point) = 0.5 * sign(signed_2area) * d(signed_2area)/d(point)
    
    grad = np.zeros((n, 2))
    for idx in active:
        i, j, k = indices[idx]
        sign = np.sign(signed_2area[idx])
        if sign == 0:
            sign = 1.0
        
        # d(signed_2area)/dp1:
        # signed_2area = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
        # d/dx1 = (y2-y3), d/dy1 = (x3-x2)
        # d/dx2 = (y3-y1), d/dy2 = (x1-x3)
        # d/dx3 = (y1-y2), d/dy3 = (x2-x1)
        
        p1_pt = points[i]
        p2_pt = points[j]
        p3_pt = points[k]
        
        # Gradient for point i
        grad[i, 0] += 0.5 * sign * (p2_pt[1] - p3_pt[1])
        grad[i, 1] += 0.5 * sign * (p3_pt[0] - p2_pt[0])
        
        # Gradient for point j
        grad[j, 0] += 0.5 * sign * (p3_pt[1] - p1_pt[1])
        grad[j, 1] += 0.5 * sign * (p1_pt[0] - p3_pt[0])
        
        # Gradient for point k
        grad[k, 0] += 0.5 * sign * (p1_pt[1] - p2_pt[1])
        grad[k, 1] += 0.5 * sign * (p2_pt[0] - p1_pt[0])
    
    return grad

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 590  # Leave 10s safety margin
    
    # Start with regular 13-gon as initial guess
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_points)
    
    # Subgradient ascent to maximize A_min / A_hull
    # We'll use a line search approach
    
    def _subgradient_ascent(points, max_iters=5000):
        """Perform subgradient ascent with adaptive step size."""
        pts = points.copy()
        best_local = pts.copy()
        best_local_score = _score(pts)
        
        # Initial step size
        step_size = 0.01
        patience = 0
        max_patience = 50
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Compute score and subgradient
            score = _score(pts)
            if score <= 0:
                break
            
            if score > best_local_score:
                best_local_score = score
                best_local = pts.copy()
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                # Reduce step size and reset patience
                step_size *= 0.5
                patience = 0
                if step_size < 1e-10:
                    break
            
            # Compute subgradient
            grad = _subgradient(pts)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            # Normalize gradient
            grad = grad / grad_norm
            
            # Try a few step sizes (backtracking line search)
            improved = False
            for factor in [1.0, 0.5, 0.25, 0.1]:
                new_step = step_size * factor
                new_pts = pts + new_step * grad
                new_score = _score(new_pts)
                if new_score > score * 0.999:  # Allow slight decrease for progress
                    pts = new_pts
                    improved = True
                    if new_score > score:
                        step_size = min(step_size * 1.2, 0.1)
                    break
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-10:
                    break
        
        return best_local, best_local_score
    
    # Run subgradient ascent from multiple starting points
    rng = np.random.default_rng(42)
    
    # Starting point 1: regular polygon
    start_points = [best_points.copy()]
    
    # Starting point 2: regular polygon with slight perturbation
    perturbed = best_points + rng.normal(0, 0.05, size=(n, 2))
    start_points.append(perturbed)
    
    # Starting point 3: different rotation
    angles2 = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
    start_points.append(np.column_stack((np.cos(angles2), np.sin(angles2))))
    
    for start in start_points:
        if time.time() > deadline - 30:
            break
        
        local_best, local_score = _subgradient_ascent(start, max_iters=3000)
        if local_score > best_score:
            best_score = local_score
            best_points = local_best
    
    # Final polish: run more iterations from the best point
    if time.time() < deadline - 30:
        local_best, local_score = _subgradient_ascent(best_points, max_iters=5000)
        if local_score > best_score:
            best_score = local_score
            best_points = local_best
    
    # Ensure the result is valid
    if _score(best_points) <= 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
