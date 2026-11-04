"""Heilbronn convex regions n=14 — Voronoi relaxation + Nelder-Mead optimization."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Voronoi
from scipy.optimize import minimize

N = 14
DEADLINE = time.time() + 130.0  # Safety margin before 150s limit


def _min_triangle_area(points):
    """Compute minimum area of any triangle formed by triples of points (vectorized)."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute area of convex hull of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _voronoi_centroid(points, disk_radius=1.0):
    """
    For each point, compute the centroid of its Voronoi cell clipped to a unit disk.
    Returns new positions as the centroids.
    """
    n = len(points)
    new_points = points.copy()
    
    try:
        vor = Voronoi(points)
    except Exception:
        return new_points
    
    # For each point, find its Voronoi cell vertices and compute centroid
    for i in range(n):
        # Get the vertices of the Voronoi cell for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Unbounded cell — keep original position or project to disk boundary
            continue
        
        # Get the vertices of this region
        vertices = vor.vertices[region]
        
        if len(vertices) < 3:
            continue
        
        # Clip to unit disk: keep only vertices inside the disk
        inside = vertices[np.sum(vertices**2, axis=1) <= disk_radius**2 * 1.01]
        
        if len(inside) < 3:
            # If too few vertices inside, use all vertices but project centroid
            centroid = np.mean(vertices, axis=0)
            # Project to disk if outside
            dist = np.linalg.norm(centroid)
            if dist > disk_radius:
                centroid = centroid / dist * disk_radius
            new_points[i] = centroid
            continue
        
        # Compute centroid of the clipped polygon
        # Use the polygon vertices in order
        centroid = np.mean(inside, axis=0)
        
        # Also consider the original point as part of the cell
        # Weighted average: centroid of cell vertices + original point
        new_points[i] = 0.7 * centroid + 0.3 * points[i]
        
        # Project to disk
        dist = np.linalg.norm(new_points[i])
        if dist > disk_radius:
            new_points[i] = new_points[i] / dist * disk_radius
    
    return new_points


def _voronoi_relax(points, n_iter=50, disk_radius=1.0):
    """Iteratively move points to Voronoi cell centroids."""
    current = points.copy()
    for _ in range(n_iter):
        if time.time() > DEADLINE:
            break
        new_pts = _voronoi_centroid(current, disk_radius)
        # Check if converged
        if np.max(np.abs(new_pts - current)) < 1e-10:
            break
        current = new_pts
    return current


def _two_ring_initial(n_outer=7, n_inner=7, r_inner=0.5, rotation=0.5):
    """Generate two-ring initial configuration."""
    points = []
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer
        points.append([np.cos(angle), np.sin(angle)])
    for i in range(n_inner):
        angle = 2 * np.pi * (i + rotation) / n_inner
        points.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
    return np.array(points, dtype=np.float64)


def _three_ring_initial(r1=1.0, r2=0.65, r3=0.35, n1=7, n2=4, n3=3):
    """Generate three-ring initial configuration."""
    points = []
    for i in range(n1):
        angle = 2 * np.pi * i / n1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    for i in range(n2):
        angle = 2 * np.pi * (i + 0.5) / n2
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    for i in range(n3):
        angle = 2 * np.pi * (i + 0.25) / n3
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    return np.array(points, dtype=np.float64)


def _single_ring_initial(n=14, r=1.0):
    """Generate single ring initial configuration."""
    points = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points, dtype=np.float64)


def _nelder_mead_optimize(points, max_iter=500):
    """Optimize using Nelder-Mead on the flattened point coordinates."""
    flat = points.reshape(-1)
    
    def obj(x):
        if time.time() > DEADLINE:
            return 1e10
        pts = x.reshape(N, 2)
        return _neg_score(pts)
    
    try:
        res = minimize(obj, flat, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if res.fun < _neg_score(points):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points


def _coordinate_descent_polish(points, max_rounds=30, step=0.01):
    """Coordinate descent: try small moves for each point."""
    best = points.copy()
    best_score = _score(best)
    
    for rnd in range(max_rounds):
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(N):
            if time.time() > DEADLINE:
                break
            for dx in [-step, 0.0, step]:
                for dy in [-step, 0.0, step]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    candidate = best.copy()
                    candidate[i, 0] += dx
                    candidate[i, 1] += dy
                    s = _score(candidate)
                    if s > best_score + 1e-12:
                        best = candidate
                        best_score = s
                        improved = True
        if not improved:
            break
    
    return best


def _apply_affine(points, angle=0.0, shear=0.0, scale=1.0):
    """Apply affine transformation: rotation + shear + scaling."""
    theta = angle
    c, s = np.cos(theta), np.sin(theta)
    # Rotation matrix
    R = np.array([[c, -s], [s, c]])
    # Shear matrix
    S = np.array([[1, shear], [0, 1]])
    # Combined
    M = scale * (R @ S)
    return points @ M.T


def construct():
    """Construct 14 points maximizing min triangle area / hull area."""
    start_time = time.time()
    
    # Generate multiple initial configurations
    inits = []
    
    # Two-ring variants
    for r_inner in [0.4, 0.5, 0.6]:
        for rot in [0.3, 0.5, 0.7]:
            inits.append(_two_ring_initial(7, 7, r_inner, rot))
    
    # Three-ring
    inits.append(_three_ring_initial(1.0, 0.65, 0.35, 7, 4, 3))
    inits.append(_three_ring_initial(1.0, 0.7, 0.4, 7, 4, 3))
    
    # Single ring
    inits.append(_single_ring_initial(14, 1.0))
    
    # Perturbed versions
    rng = np.random.RandomState(42)
    for base in inits[:4]:
        perturbed = base + 0.03 * rng.randn(N, 2)
        inits.append(perturbed)
    
    # Evaluate all initial configurations
    best_points = None
    best_score = -1.0
    
    for init in inits:
        if time.time() > DEADLINE:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()
    
    if best_points is None:
        best_points = _two_ring_initial(7, 7, 0.5, 0.5)
        best_score = _score(best_points)
    
    # Apply Voronoi relaxation to the best initial config
    if time.time() < DEADLINE - 60:
        voronoi_pts = _voronoi_relax(best_points, n_iter=80, disk_radius=1.0)
        voronoi_score = _score(voronoi_pts)
        if voronoi_score > best_score:
            best_points = voronoi_pts
            best_score = voronoi_score
    
    # Try Voronoi on other good inits too
    for init in inits[:6]:
        if time.time() > DEADLINE - 40:
            break
        vor = _voronoi_relax(init, n_iter=40, disk_radius=1.0)
        s = _score(vor)
        if s > best_score:
            best_score = s
            best_points = vor
    
    # Nelder-Mead optimization from best
    if time.time() < DEADLINE - 40:
        nm_pts = _nelder_mead_optimize(best_points, max_iter=300)
        nm_score = _score(nm_pts)
        if nm_score > best_score:
            best_points = nm_pts
            best_score = nm_score
    
    # Coordinate descent polish
    if time.time() < DEADLINE - 20:
        polished = _coordinate_descent_polish(best_points, max_rounds=40, step=0.005)
        pol_score = _score(polished)
        if pol_score > best_score:
            best_points = polished
            best_score = pol_score
    
    # Try affine transformations to break symmetry
    if time.time() < DEADLINE - 15:
        for angle in [0.1, 0.2, 0.3]:
            for shear in [0.05, 0.1]:
                transformed = _apply_affine(best_points, angle=angle, shear=shear)
                t_score = _score(transformed)
                if t_score > best_score:
                    best_score = t_score
                    best_points = transformed.copy()
    
    # Final Nelder-Mead polish
    if time.time() < DEADLINE - 10:
        final_nm = _nelder_mead_optimize(best_points, max_iter=200)
        final_score = _score(final_nm)
        if final_score > best_score:
            best_points = final_nm
            best_score = final_score
    
    # Final coordinate descent
    if time.time() < DEADLINE - 5:
        final_cd = _coordinate_descent_polish(best_points, max_rounds=20, step=0.003)
        final_cd_score = _score(final_cd)
        if final_cd_score > best_score:
            best_points = final_cd
            best_score = final_cd_score
    
    return best_points
