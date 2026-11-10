# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])) / 2.0


def _all_triangle_areas(points):
    """Compute all triangle areas among all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(points[i], points[j], points[k])
                areas.append(a)
    return np.array(areas)


def _min_triangle_area(points):
    """Compute minimum triangle area."""
    return np.min(_all_triangle_areas(points))


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute A_min / A_hull."""
    try:
        a_min = _min_triangle_area(points)
        a_hull = _hull_area(points)
        if a_hull <= 0 or a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0


def _laplacian_repulsion(points, n_iter=200, alpha=0.1, repulse_strength=0.01):
    """
    Physics-based relaxation: Laplacian smoothing + pairwise repulsion.
    Moves points toward even spatial distribution.
    """
    n = len(points)
    pts = points.copy().astype(float)
    
    for it in range(n_iter):
        new_pts = pts.copy()
        
        # Laplacian smoothing: move each point toward centroid of neighbors
        # Use Delaunay neighbors if possible, else all other points
        try:
            tri = Delaunay(pts)
            # For each point, find its neighbors via Delaunay
            for i in range(n):
                # Find triangles containing point i
                neighbors = set()
                for simplex in tri.simplices:
                    if i in simplex:
                        for j in simplex:
                            if j != i:
                                neighbors.add(j)
                if neighbors:
                    centroid = np.mean(pts[list(neighbors)], axis=0)
                    new_pts[i] = pts[i] + alpha * (centroid - pts[i])
                else:
                    # Fallback: average of all other points
                    others = [pts[j] for j in range(n) if j != i]
                    centroid = np.mean(others, axis=0)
                    new_pts[i] = pts[i] + alpha * (centroid - pts[i])
        except Exception:
            # Fallback: simple average of all other points
            for i in range(n):
                others = [pts[j] for j in range(n) if j != i]
                centroid = np.mean(others, axis=0)
                new_pts[i] = pts[i] + alpha * (centroid - pts[i])
        
        # Repulsive forces: push points apart to avoid clustering
        for i in range(n):
            for j in range(i+1, n):
                d = new_pts[i] - new_pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    d = np.array([1e-8, 0.0])
                    dist = 1e-8
                # Repulsion force ~ 1/dist^2, scaled
                force = repulse_strength / (dist * dist) * d / dist
                new_pts[i] += force
                new_pts[j] -= force
        
        # Normalize: keep points in [0,1]^2 roughly
        # Re-center and scale
        center = np.mean(new_pts, axis=0)
        new_pts -= center
        max_coord = np.max(np.abs(new_pts))
        if max_coord > 0:
            new_pts = new_pts / max_coord * 0.45 + 0.5
        
        pts = new_pts
    
    return pts


def _nelder_mead_optimize(points, max_iter=500):
    """
    Optimize points using Nelder-Mead on the negative score.
    We maximize score = A_min / A_hull, so minimize -score.
    """
    n = len(points)
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        s = _score(pts)
        return -s
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    return result.x.reshape(n, 2)


def _init_regular_polygon(n):
    """Regular n-gon on unit circle."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _init_grid_plus_center(n):
    """Grid-like arrangement with center point."""
    # For 13 points: 3x4 grid = 12 + 1 center
    if n == 13:
        xs = np.array([0.2, 0.5, 0.8])
        ys = np.array([0.2, 0.4, 0.6, 0.8])
        pts = []
        for x in xs:
            for y in ys:
                pts.append([x, y])
        pts.append([0.5, 0.5])
        pts = np.array(pts)
        # Add slight jitter to break symmetry
        rng = np.random.default_rng(123)
        pts += rng.normal(0, 0.01, pts.shape)
        return pts
    else:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])


def _init_random(n, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def _init_fibonacci_sphere_projected(n):
    """Fibonacci lattice projected to 2D disk."""
    golden = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        # Fibonacci sphere
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(1 - z*z)
        theta = 2 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    return np.array(points)


def _init_concentric(n):
    """Concentric rings."""
    points = []
    # Center
    points.append([0.5, 0.5])
    # Inner ring: 4 points
    for i in range(4):
        angle = 2 * np.pi * i / 4
        points.append([0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)])
    # Outer ring: 8 points
    for i in range(8):
        angle = 2 * np.pi * i / 8
        points.append([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
    return np.array(points[:n])


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin before 630s kill
    
    # Multiple initialization strategies
    inits = []
    inits.append(('polygon', _init_regular_polygon(n)))
    inits.append(('grid', _init_grid_plus_center(n)))
    inits.append(('fibonacci', _init_fibonacci_sphere_projected(n)))
    inits.append(('concentric', _init_concentric(n)))
    for s in [1, 2, 3, 42, 123, 777]:
        inits.append((f'rand_{s}', _init_random(n, s)))
    
    best_points = None
    best_score = -np.inf
    
    for name, init_pts in inits:
        if time.time() > deadline:
            break
        
        # Step 1: Physics-based relaxation
        relaxed = _laplacian_repulsion(init_pts, n_iter=150, alpha=0.08, repulse_strength=0.005)
        
        # Evaluate after relaxation
        s_relaxed = _score(relaxed)
        if s_relaxed > best_score:
            best_score = s_relaxed
            best_points = relaxed.copy()
        
        # Step 2: Nelder-Mead polishing
        if time.time() > deadline - 60:
            break
        
        try:
            polished = _nelder_mead_optimize(relaxed, max_iter=300)
            s_polished = _score(polished)
            if s_polished > best_score:
                best_score = s_polished
                best_points = polished.copy()
        except Exception:
            pass
    
    # Final check
    if best_points is None:
        best_points = _init_regular_polygon(n)
    
    # Ensure finite and valid
    if not np.all(np.isfinite(best_points)):
        best_points = _init_regular_polygon(n)
    
    # Verify convex hull works
    try:
        ConvexHull(best_points)
    except Exception:
        best_points = _init_regular_polygon(n)
    
    return best_points


# EVOLVE-BLOCK-END
