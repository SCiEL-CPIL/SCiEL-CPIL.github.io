"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11
# Equilateral triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the container triangle

def _bary_to_cart(b):
    """Convert barycentric coords (alpha, beta, gamma) with alpha+beta+gamma=1 to Cartesian."""
    # b[0]=alpha (at A), b[1]=beta (at B), b[2]=gamma (at C)
    return b[0] * A + b[1] * B + b[2] * C

def _cart_to_bary(p):
    """Convert Cartesian to barycentric (alpha, beta, gamma)."""
    alpha = 1.0 - p[0] - p[1] / (np.sqrt(3) / 2)
    beta = p[0]
    gamma = p[1] / (np.sqrt(3) / 2)
    return np.array([alpha, beta, gamma])

def _all_triangle_areas(points):
    """Compute areas of all triangles formed by pairs of points. Returns array of areas (absolute, doubled for speed)."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                # 2*area = |cross(b-a, c-a)|
                cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                areas.append(abs(cross) * 0.5)
    return np.array(areas)

def _min_area(points):
    """Minimum triangle area among all triples."""
    areas = _all_triangle_areas(points)
    return np.min(areas)

def _neg_min_area(flat):
    """Objective: negative of min triangle area, with penalty for points outside triangle."""
    n = N
    pts = flat.reshape(n, 2)
    # Check if all points are inside the triangle using barycentric coords
    # For point p, barycentric: alpha = 1 - p0 - p1/(h), beta = p0, gamma = p1/h where h = sqrt(3)/2
    h = np.sqrt(3) / 2
    alpha = 1.0 - pts[:, 0] - pts[:, 1] / h
    beta = pts[:, 0]
    gamma = pts[:, 1] / h
    # All must be >= 0 for point to be inside (with tolerance)
    min_bary = np.min(np.column_stack([alpha, beta, gamma]), axis=1)
    if np.any(min_bary < -1e-9):
        return 1e10  # large penalty
    # Clamp tiny negatives to 0
    pts_clamped = pts.copy()
    # Compute min area
    areas = _all_triangle_areas(pts_clamped)
    return -np.min(areas)

def _symmetric_initial():
    """Generate a symmetric initial configuration for 11 points."""
    # Known good structure: center point, ring of 6, ring of 4 (or similar)
    # For 11 points, a good symmetric layout:
    # - 1 center
    # - 6 points on a hexagonal ring
    # - 4 points on outer ring (at midpoints of edges)
    
    h = np.sqrt(3) / 2
    center = np.array([0.5, h / 3.0])  # centroid
    
    points = [center]
    
    # Ring of 6: at distance r1 from center, at angles 0, 60, 120, 180, 240, 300
    r1 = 0.25
    for k in range(6):
        angle = k * np.pi / 3
        x = center[0] + r1 * np.cos(angle)
        y = center[1] + r1 * np.sin(angle)
        points.append([x, y])
    
    # 4 more points: at midpoints of the three edges + one more
    # Midpoints of edges
    mid_ab = (A + B) / 2
    mid_bc = (B + C) / 2
    mid_ca = (C + A) / 2
    points.append(mid_ab)
    points.append(mid_bc)
    points.append(mid_ca)
    
    # That's 10, need 1 more - put it at center offset
    points.append([0.5, h * 2 / 3.0])  # top area
    
    return np.array(points)

def construct():
    deadline = time.time() + 130.0  # 130s safety margin before 150s kill
    
    best_points = _symmetric_initial()
    best_val = _min_area(best_points)
    
    # Try differential evolution first (global optimization)
    # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
    h = np.sqrt(3) / 2
    bounds = [(0, 1), (0, h)] * N
    
    # Use a smaller population for speed
    try:
        result = differential_evolution(
            _neg_min_area,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            polish=True,
            callback=None
        )
        if time.time() < deadline - 10:
            pts = result.x.reshape(N, 2)
            # Verify inside triangle
            alpha = 1.0 - pts[:, 0] - pts[:, 1] / h
            beta = pts[:, 0]
            gamma = pts[:, 1] / h
            if np.all(alpha >= -1e-8) and np.all(beta >= -1e-8) and np.all(gamma >= -1e-8):
                val = _min_area(pts)
                if val > best_val:
                    best_points = pts
                    best_val = val
    except Exception:
        pass
    
    # Local refinement with L-BFGS-B
    if time.time() < deadline - 15:
        try:
            # Start from best found
            x0 = best_points.flatten()
            res = minimize(
                _neg_min_area,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() < deadline - 5:
                pts = res.x.reshape(N, 2)
                alpha = 1.0 - pts[:, 0] - pts[:, 1] / h
                beta = pts[:, 0]
                gamma = pts[:, 1] / h
                if np.all(alpha >= -1e-8) and np.all(beta >= -1e-8) and np.all(gamma >= -1e-8):
                    val = _min_area(pts)
                    if val > best_val:
                        best_points = pts
                        best_val = val
        except Exception:
            pass
    
    # Try a few random restarts with local optimization if time permits
    rng = np.random.RandomState(123)
    while time.time() < deadline - 20:
        # Random symmetric-ish start
        pts = _symmetric_initial()
        # Add small random perturbation
        pts = pts + rng.uniform(-0.02, 0.02, size=pts.shape)
        # Clip to valid range
        pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, h)
        
        x0 = pts.flatten()
        try:
            res = minimize(
                _neg_min_area,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() < deadline - 5:
                pts2 = res.x.reshape(N, 2)
                alpha = 1.0 - pts2[:, 0] - pts2[:, 1] / h
                beta = pts2[:, 0]
                gamma = pts2[:, 1] / h
                if np.all(alpha >= -1e-8) and np.all(beta >= -1e-8) and np.all(gamma >= -1e-8):
                    val = _min_area(pts2)
                    if val > best_val:
                        best_points = pts2
                        best_val = val
        except Exception:
            pass
    
    return best_points
