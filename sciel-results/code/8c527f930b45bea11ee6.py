# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time


def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3 (each shape (2,))."""
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))


def _min_triangle_area(points):
    """Minimum area among all C(13,3)=286 triangles."""
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
    """Area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _objective(params, points_init, hull_indices):
    """
    Objective to minimize: -A_min / A_hull.
    params: flattened array of point coordinates.
    """
    points = params.reshape(-1, 2)
    
    # Check for NaN or inf
    if not np.all(np.isfinite(points)):
        return 1e10
    
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    
    if hull_area <= 1e-12 or min_area <= 1e-12:
        return 1e10
    
    return -(min_area / hull_area)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize
    the ratio A_min / A_hull, where A_min is the smallest triangle area and A_hull
    is the convex hull area.
    
    Strategy:
    1. Start with a regular 13-gon (all points on hull).
    2. Move some points inward and use Lloyd's algorithm (Voronoi centroid relaxation)
       to improve the distribution.
    3. Perform local optimization using scipy.optimize.minimize to maximize min triangle area.
    4. Use a deadline guard to ensure we return within time limits.
    
    Returns:
        points: np.ndarray of shape (13,2)
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 seconds total
    
    n = 13
    
    # Initial configuration: regular 13-gon
    angles = 2 * np.pi * np.arange(n) / n
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Scale to unit area or keep as is
    best_points = points.copy()
    best_score = -_objective(points.flatten(), points, np.arange(n))
    
    # Try several different initial configurations
    configs = []
    
    # Config 1: Regular 13-gon
    configs.append(points.copy())
    
    # Config 2: Regular 13-gon with some points moved toward center
    for scale in [0.8, 0.6, 0.5, 0.4, 0.3]:
        p = points.copy()
        # Move interior points (indices 1-11) toward center
        for i in range(1, 12):
            p[i] = p[i] * scale
        configs.append(p.copy())
    
    # Config 3: Hexagonal-like pattern
    # Outer ring: 6 points, inner ring: 6 points, center: 1 point
    p3 = np.zeros((13, 2))
    # Outer hexagon
    for i in range(6):
        angle = 2 * np.pi * i / 6
        p3[i] = [np.cos(angle), np.sin(angle)]
    # Inner hexagon (rotated)
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        p3[6 + i] = 0.5 * [np.cos(angle), np.sin(angle)]
    # Center
    p3[12] = [0.0, 0.0]
    configs.append(p3.copy())
    
    # Config 4: Two concentric rings
    p4 = np.zeros((13, 2))
    for i in range(7):
        angle = 2 * np.pi * i / 7
        p4[i] = [np.cos(angle), np.sin(angle)]
    for i in range(5):
        angle = 2 * np.pi * i / 5 + 0.3
        p4[7 + i] = 0.55 * [np.cos(angle), np.sin(angle)]
    p4[12] = [0.0, 0.0]
    configs.append(p4.copy())
    
    # Config 5: Random with Lloyd relaxation
    rng = np.random.default_rng(seed=42)
    p5 = rng.random((13, 2))
    configs.append(p5.copy())
    
    # Evaluate each config and keep the best
    for cfg in configs:
        if time.time() > deadline - 60:
            break
        score = -_objective(cfg.flatten(), cfg, np.arange(n))
        if score > best_score:
            best_score = score
            best_points = cfg.copy()
    
    # Now perform local optimization on the best config
    if time.time() < deadline - 120:
        # Use Nelder-Mead for the optimization
        x0 = best_points.flatten()
        result = minimize(
            _objective,
            x0,
            args=(best_points, np.arange(n)),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result.success or result.fun < -best_score:
            opt_points = result.x.reshape(-1, 2)
            if np.all(np.isfinite(opt_points)):
                opt_score = -result.fun
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
    
    # Try a second round of optimization with different initial points if time permits
    if time.time() < deadline - 60:
        # Generate a few more random starting points
        for seed in [1, 2, 3]:
            if time.time() > deadline - 30:
                break
            rng2 = np.random.default_rng(seed=seed)
            p_rand = rng2.random((13, 2))
            
            # Simple Lloyd relaxation: 50 iterations
            for it in range(50):
                if time.time() > deadline - 30:
                    break
                try:
                    delaunay = Delaunay(p_rand)
                    # Compute Voronoi centroids (approximate via Delaunay)
                    for i in range(13):
                        # Find neighboring points in Delaunay
                        neighbors = delaunay.simplices[delaunay.simplices == i]
                        if len(neighbors) > 0:
                            # Average of neighboring points as approximation
                            centroid = np.mean(neighbors, axis=0)
                            p_rand[i] = 0.5 * p_rand[i] + 0.5 * centroid
                except Exception:
                    break
            
            # Optimize
            score = -_objective(p_rand.flatten(), p_rand, np.arange(n))
            if score > best_score and time.time() < deadline - 30:
                result2 = minimize(
                    _objective,
                    p_rand.flatten(),
                    args=(p_rand, np.arange(n)),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result2.success or result2.fun < -best_score:
                    opt_points2 = result2.x.reshape(-1, 2)
                    if np.all(np.isfinite(opt_points2)):
                        opt_score2 = -result2.fun
                        if opt_score2 > best_score:
                            best_score = opt_score2
                            best_points = opt_points2.copy()
    
    # Final check: ensure valid output
    if not np.all(np.isfinite(best_points)) or len(best_points) != 13:
        # Fallback to regular 13-gon
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points


# EVOLVE-BLOCK-END
