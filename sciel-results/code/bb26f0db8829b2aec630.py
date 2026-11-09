# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

# Deadline guard: keep a safety margin before the 630s wall-clock limit
_DEADLINE = 630.0 - 30.0  # 600s effective budget
_START_TIME = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas vectorized. Returns (n_tri, 3) indices and areas."""
    n = len(points)
    idx_i, idx_j, idx_k = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx_i.append(i)
                idx_j.append(j)
                idx_k.append(k)
    
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    
    return idx_i, idx_j, idx_k, areas


def _score(points):
    """Compute A_min / A_hull for the given points."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area <= 1e-15:
        return 0.0
    
    _, _, _, areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area


def _subgradient(points, idx_i, idx_j, idx_k, areas):
    """Compute subgradient of the minimum triangle area.
    
    Returns gradient w.r.t. each point coordinate (n, 2) array.
    The subgradient of min is the average gradient of the active (minimum) triangles.
    """
    n = len(points)
    grad = np.zeros((n, 2))
    
    min_area = np.min(areas)
    # Find all triangles with area close to minimum (within tolerance)
    tol = max(1e-10, 1e-6 * min_area)
    active = np.where(areas <= min_area + tol)[0]
    
    if len(active) == 0:
        return grad
    
    # For each active triangle, compute gradient of its area w.r.t. point coordinates
    # Area of triangle (p1, p2, p3) = 0.5 * |cross(p2-p1, p3-p1)|
    # d(Area)/dp1 = 0.5 * cross(p3-p2, n) where n is normal... 
    # Actually, let's use the formula:
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # d(Area)/dx1 = 0.5 * (y2 - y3)
    # d(Area)/dy1 = 0.5 * (x3 - x2)
    # d(Area)/dx2 = 0.5 * (y3 - y1)
    # d(Area)/dy2 = 0.5 * (x1 - x3)
    # d(Area)/dx3 = 0.5 * (y1 - y2)
    # d(Area)/dy3 = 0.5 * (x2 - x1)
    
    for tri_idx in active:
        i = idx_i[tri_idx]
        j = idx_j[tri_idx]
        k = idx_k[tri_idx]
        
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Signed area (without 0.5 and abs)
        signed = (p1[0] * (p2[1] - p3[1]) +
                  p2[0] * (p3[1] - p1[1]) +
                  p3[0] * (p1[1] - p2[1]))
        
        # Gradient of |signed| = sign(signed) * gradient of signed
        sign = np.sign(signed)
        if sign == 0:
            sign = 1.0
        
        # Gradient of signed area w.r.t. each point
        # d(signed)/dx1 = (y2 - y3), d(signed)/dy1 = (x3 - x2)
        # d(signed)/dx2 = (y3 - y1), d(signed)/dy2 = (x1 - x3)
        # d(signed)/dx3 = (y1 - y2), d(signed)/dy3 = (x2 - x1)
        
        grad[i, 0] += sign * (p2[1] - p3[1])
        grad[i, 1] += sign * (p3[0] - p2[0])
        grad[j, 0] += sign * (p3[1] - p1[1])
        grad[j, 1] += sign * (p1[0] - p3[0])
        grad[k, 0] += sign * (p1[1] - p2[1])
        grad[k, 1] += sign * (p2[0] - p1[0])
    
    # Normalize by number of active triangles
    grad /= len(active)
    
    return grad


def _optimize_subgradient(points, max_iter=20000, lr_init=0.002, lr_decay=0.9999):
    """Optimize using subgradient ascent on minimum triangle area."""
    points = points.copy().astype(float)
    n = len(points)
    
    # Precompute triangle indices
    idx_i, idx_j, idx_k, areas = _triangle_areas_vectorized(points)
    
    best_score = _score(points)
    best_points = points.copy()
    
    lr = lr_init
    no_improve_count = 0
    
    for it in range(max_iter):
        if _time_left() < 30:
            break
        
        # Compute current state
        idx_i, idx_j, idx_k, areas = _triangle_areas_vectorized(points)
        min_area = np.min(areas)
        
        try:
            hull_area = ConvexHull(points).volume
        except Exception:
            break
        if hull_area <= 1e-15:
            break
        
        score = min_area / hull_area
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Compute subgradient
        grad = _subgradient(points, idx_i, idx_j, idx_k, areas)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        # Update points
        points += lr * grad
        
        # Keep points within reasonable bounds
        points = np.clip(points, -0.1, 1.1)
        
        # Adaptive learning rate
        if no_improve_count > 500:
            lr *= 0.95
            no_improve_count = 0
        elif it % 1000 == 0:
            lr = max(1e-5, lr * lr_decay)
        
        # Check for degenerate cases
        if not np.all(np.isfinite(points)):
            points = best_points.copy()
            break
    
    return best_points


def _local_search_refine(points, n_points, max_iter=500):
    """Refine using local search with random perturbations."""
    points = points.copy().astype(float)
    n = n_points
    
    best_score = _score(points)
    best_points = points.copy()
    
    rng = np.random.default_rng(seed=42)
    
    for it in range(max_iter):
        if _time_left() < 30:
            break
        
        improved = False
        
        for pt_idx in range(n):
            if _time_left() < 30:
                break
            
            original = points[pt_idx].copy()
            
            # Try several random perturbations with decreasing step sizes
            for step_size in [0.02, 0.01, 0.005, 0.002]:
                for attempt in range(5):
                    if _time_left() < 30:
                        break
                    
                    perturbation = rng.normal(0, step_size, size=2)
                    test_points = points.copy()
                    test_points[pt_idx] = original + perturbation
                    
                    if not np.all(np.isfinite(test_points)):
                        continue
                    if np.any(test_points < -0.5) or np.any(test_points > 1.5):
                        continue
                    
                    test_score = _score(test_points)
                    if test_score > best_score:
                        best_score = test_score
                        best_points = test_points.copy()
                        points = test_points.copy()
                        improved = True
                        break
                
                if improved:
                    break
        
        if not improved:
            break
    
    return best_points


def _init_regular_polygon():
    """Initialize with regular polygon pattern."""
    n = 13
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.array([0.45 if i % 2 == 0 else 0.3 for i in range(n)])
    
    points = np.column_stack([
        0.5 + radii * np.cos(angles),
        0.5 + radii * np.sin(angles)
    ])
    return points


def _init_halton():
    """Initialize using Halton sequence for good distribution."""
    n = 13
    points = []
    
    # Halton base 2 (x-coordinate)
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        points.append([0.1 + 0.8 * x, 0.1 + 0.8 * y])
    
    return np.array(points)


def _init_concentric():
    """Initialize with concentric rings."""
    # 4 outer, 5 middle, 4 inner
    angles_outer = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    angles_mid = np.linspace(np.pi / 5, 2 * np.pi, 5, endpoint=False)
    angles_inner = np.linspace(np.pi / 4, 2 * np.pi, 4, endpoint=False)
    
    outer_pts = np.column_stack([
        0.5 + 0.45 * np.cos(angles_outer),
        0.5 + 0.45 * np.sin(angles_outer)
    ])
    mid_pts = np.column_stack([
        0.5 + 0.28 * np.cos(angles_mid),
        0.5 + 0.28 * np.sin(angles_mid)
    ])
    inner_pts = np.column_stack([
        0.5 + 0.12 * np.cos(angles_inner),
        0.5 + 0.12 * np.sin(angles_inner)
    ])
    
    return np.vstack([outer_pts, mid_pts, inner_pts])


def _init_fibonacci():
    """Initialize using Fibonacci distribution."""
    n = 13
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = []
    
    for i in range(n):
        angle = i * golden_angle
        r = 0.5 * np.sqrt((i + 0.5) / n)
        points.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    
    return np.array(points)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    best_points = None
    best_score = -1
    
    # Generate several initial candidates
    candidates = []
    
    # Candidate 1: Regular polygon-based
    candidates.append(_init_regular_polygon())
    
    # Candidate 2: Halton sequence
    candidates.append(_init_halton())
    
    # Candidate 3: Concentric rings
    candidates.append(_init_concentric())
    
    # Candidate 4: Fibonacci distribution
    candidates.append(_init_fibonacci())
    
    # Candidate 5: Random
    rng = np.random.default_rng(seed=42)
    candidates.append(0.1 + 0.8 * rng.random((n, 2)))
    
    # Candidate 6: Another random
    rng2 = np.random.default_rng(seed=123)
    candidates.append(0.1 + 0.8 * rng2.random((n, 2)))
    
    # Optimize each candidate
    for candidate in candidates:
        if _time_left() < 30:
            break
        
        # Subgradient optimization
        optimized = _optimize_subgradient(candidate, max_iter=10000)
        score = _score(optimized)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
        
        # Local search refinement if we have time
        if _time_left() > 60:
            refined = _local_search_refine(optimized, n, max_iter=300)
            score2 = _score(refined)
            if score2 > best_score:
                best_score = score2
                best_points = refined.copy()
    
    # Final fallback: if all failed, return the best we have
    if best_points is None:
        best_points = candidates[0]
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = np.random.default_rng(0).random((n, 2))
    
    return best_points


# EVOLVE-BLOCK-END
