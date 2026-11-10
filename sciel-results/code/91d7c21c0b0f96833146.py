# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])) / 2.0


def _min_triangle_area(points):
    """Compute minimum triangle area among all C(n,3) triples."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _physics_relax(points, hull_points, n_iter=200, dt=0.01):
    """
    Physics-based relaxation:
    - Laplacian smoothing: each point moves toward centroid of neighbors
    - Repulsive forces: points repel each other to avoid clustering
    - Constrain to stay inside convex hull
    """
    n = len(points)
    pts = points.copy()
    
    # Precompute convex hull for containment check
    try:
        hull = ConvexHull(hull_points)
        hull_verts = hull_points[hull.vertices]
    except Exception:
        return pts
    
    # Build neighbor graph based on proximity (k-nearest)
    k = min(6, n - 1)
    
    for it in range(n_iter):
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Laplacian smoothing force (toward centroid of k-nearest neighbors)
        for i in range(n):
            dists = np.linalg.norm(pts - pts[i], axis=1)
            dists[i] = np.inf
            neighbors = np.argsort(dists)[:k]
            centroid = pts[neighbors].mean(axis=0)
            forces[i] += (centroid - pts[i]) * 0.1
        
        # Repulsive forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-6:
                    d = np.array([1e-3, 1e-3])
                    dist = np.linalg.norm(d)
                # Repulsion strength ~ 1/d^2
                repulsion = 0.05 / (dist * dist)
                force = d / dist * repulsion
                forces[i] += force
                forces[j] -= force
        
        # Update positions
        pts = pts + dt * forces
        
        # Project back inside convex hull (simple: clamp to bounding box of hull)
        hull_min = hull_points.min(axis=0)
        hull_max = hull_points.max(axis=0)
        pts = np.clip(pts, hull_min + 0.01, hull_max - 0.01)
    
    return pts


def _nelder_mead_polish(points, hull_points, max_iter=500):
    """Polish with Nelder-Mead to maximize min_triangle_area / hull_area."""
    n = len(points)
    
    def objective(x):
        pts = x.reshape(n, 2)
        # Check all points are finite
        if not np.all(np.isfinite(pts)):
            return 1e10
        # Check no three points are collinear (min area > 0)
        a_min = _min_triangle_area(pts)
        if a_min < 1e-10:
            return 1e10
        a_hull = _hull_area(pts)
        if a_hull < 1e-10:
            return 1e10
        # We want to maximize A_min/A_hull, so minimize negative
        return -a_min / a_hull
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if result.success or result.fun < 1e9:
        return result.x.reshape(n, 2)
    return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region to maximize
    the ratio A_min / A_hull where A_min is the smallest triangle area
    and A_hull is the convex hull area.
    """
    start_time = time.time()
    deadline = 570  # seconds, with safety margin
    
    n = 13
    best_score = 0.0
    best_points = None
    
    # Try multiple initial configurations
    configs = []
    
    # Config 1: 12 points on unit circle + 1 center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    config1 = np.vstack([circle_pts, center])
    configs.append(config1)
    
    # Config 2: Slightly perturbed circle
    rng = np.random.default_rng(seed=123)
    angles2 = angles + rng.uniform(-0.1, 0.1, size=12)
    radii2 = 1.0 + rng.uniform(-0.05, 0.05, size=12)
    circle_pts2 = np.column_stack([radii2 * np.cos(angles2), radii2 * np.sin(angles2)])
    center2 = np.array([[0.0, 0.0]])
    config2 = np.vstack([circle_pts2, center2])
    configs.append(config2)
    
    # Config 3: Two concentric rings (8 outer, 4 inner, 1 center)
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    center3 = np.array([[0.0, 0.0]])
    config3 = np.vstack([outer_pts, inner_pts, center3])
    configs.append(config3)
    
    # Config 4: Random but structured
    rng4 = np.random.default_rng(seed=456)
    config4 = rng4.random((n, 2))
    # Normalize to unit square
    config4 = (config4 - config4.min(axis=0)) / (config4.max(axis=0) - config4.min(axis=0) + 1e-10)
    configs.append(config4)
    
    # Config 5: Regular 13-gon (all on convex hull)
    angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    config5 = np.column_stack([np.cos(angles13), np.sin(angles13)])
    configs.append(config5)
    
    # Config 6: 12 on ellipse + 1 center
    angles_e = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ellipse_pts = np.column_stack([1.2 * np.cos(angles_e), 0.8 * np.sin(angles_e)])
    center_e = np.array([[0.0, 0.0]])
    config6 = np.vstack([ellipse_pts, center_e])
    configs.append(config6)
    
    for ci, config in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Use the config itself as hull reference (or slightly expanded)
        hull_ref = config.copy()
        
        # Step 1: Physics relaxation
        relaxed = _physics_relax(config, hull_ref, n_iter=150, dt=0.01)
        
        # Step 2: Nelder-Mead polish
        polished = _nelder_mead_polish(relaxed, hull_ref, max_iter=300)
        
        # Step 3: Evaluate
        score = _score(polished)
        
        if score > best_score:
            best_score = score
            best_points = polished.copy()
        
        # Also try polishing the original config directly
        if time.time() < deadline - 60:
            polished2 = _nelder_mead_polish(config, hull_ref, max_iter=200)
            score2 = _score(polished2)
            if score2 > best_score:
                best_score = score2
                best_points = polished2.copy()
    
    # If we still have time, do additional refinement on the best
    if best_points is not None and time.time() < deadline - 30:
        # One more round of polish with more iterations
        hull_ref = best_points.copy()
        final = _nelder_mead_polish(best_points, hull_ref, max_iter=400)
        final_score = _score(final)
        if final_score > best_score:
            best_score = final_score
            best_points = final
    
    # Final safety check
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback: regular 13-gon
        angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(angles13), np.sin(angles13)])
    
    return best_points


# EVOLVE-BLOCK-END
