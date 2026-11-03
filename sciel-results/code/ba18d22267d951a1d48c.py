"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3 = 3 ** 0.5
H = S3 / 2.0  # height of the triangle

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def _in_triangle(p):
    """Check if point p is inside the triangle (0,0),(1,0),(0.5,H)."""
    # Barycentric coordinates
    v0 = V2 - V0
    v1 = V1 - V0
    v2 = p - V0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return False
    v = (d11 * d02 - d01 * d12) / denom
    w = (d00 * d12 - d01 * d02) / denom
    return (v >= -1e-9) and (w >= -1e-9) and (v + w <= 1.0 + 1e-9)

def _project_to_triangle(p):
    """Project point p to the interior of the triangle."""
    if _in_triangle(p):
        return p
    # Project to nearest edge or vertex
    def dist_to_segment(a, b, p):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return proj, np.linalg.norm(p - proj)
    
    best_proj = p
    best_dist = np.inf
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    for a, b in edges:
        proj, d = dist_to_segment(a, b, p)
        if d < best_dist:
            best_dist = d
            best_proj = proj
    # Move slightly inside
    centroid = (V0 + V1 + V2) / 3.0
    best_proj = 0.95 * best_proj + 0.05 * centroid
    return best_proj

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _objective(flat_points):
    """Objective function for optimization: negative of min triangle area."""
    points = flat_points.reshape(N, 2)
    # Project each point to triangle
    points = np.array([_project_to_triangle(p) for p in points])
    min_area = _min_triangle_area(points)
    return -min_area

def _make_initial_config(strategy):
    """Generate initial configurations using different strategies."""
    if strategy == 0:
        # Grid-like placement
        points = []
        # Place points in a rough grid pattern
        rows = [
            [0.1, 0.3, 0.5, 0.7, 0.9],  # bottom row
            [0.2, 0.4, 0.6, 0.8],        # middle
            [0.3, 0.5, 0.7],             # upper
            [0.4, 0.6],                   # top
            [0.5]                         # very top
        ]
        heights = [0.05, 0.2, 0.35, 0.5, 0.65]
        for i, (row, h) in enumerate(zip(rows, heights)):
            for x in row:
                # Adjust for triangle shape
                max_x = 0.5 + 0.5 * (1 - h / H)
                min_x = 0.5 - 0.5 * (1 - h / H)
                x = min_x + (x - 0.1) / 0.8 * (max_x - min_x) * 0.8 + 0.1 * (max_x - min_x)
                x = np.clip(x, min_x + 0.01, max_x - 0.01)
                points.append([x, h])
        # Take first 11
        points = points[:N]
        return np.array(points)
    elif strategy == 1:
        # Fibonacci-like spiral
        points = []
        cx, cy = 0.5, H / 3.0
        for i in range(N):
            angle = i * 2.399963229728653  # golden angle
            r = 0.35 * np.sqrt((i + 0.5) / N)
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            points.append(_project_to_triangle(np.array([x, y])))
        return np.array(points)
    elif strategy == 2:
        # Random with rejection
        np.random.seed(42 + strategy)
        points = []
        while len(points) < N:
            u, v = np.random.random(), np.random.random()
            if u + v < 1.0:
                p = np.array([u + 0.5 * v, H * v])
                # Check distance from existing points
                if all(np.linalg.norm(p - q) > 0.08 for q in points):
                    points.append(p)
        return np.array(points)
    elif strategy == 3:
        # Symmetric pattern
        points = []
        # Center
        points.append([0.5, H / 3.0])
        # Ring 1
        for angle in [0, 60, 120, 180, 240, 300]:
            rad = np.radians(angle)
            x = 0.5 + 0.25 * np.cos(rad)
            y = H / 3.0 + 0.25 * np.sin(rad)
            p = _project_to_triangle(np.array([x, y]))
            points.append(p)
        # Ring 2
        for angle in [30, 150, 270]:
            rad = np.radians(angle)
            x = 0.5 + 0.4 * np.cos(rad)
            y = H / 3.0 + 0.4 * np.sin(rad)
            p = _project_to_triangle(np.array([x, y]))
            points.append(p)
        points = points[:N]
        return np.array(points)
    else:
        # Random
        np.random.seed(100 + strategy)
        points = []
        while len(points) < N:
            u, v = np.random.random(), np.random.random()
            if u + v < 0.95:
                p = np.array([u + 0.5 * v, H * v])
                points.append(p)
        return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second deadline with safety margin
    
    best_points = None
    best_area = -np.inf
    
    # Try multiple initial strategies
    for strategy in range(8):
        if time.time() > deadline - 10.0:
            break
        
        initial = _make_initial_config(strategy)
        flat_initial = initial.flatten()
        
        # Optimize using SLSQP
        try:
            result = minimize(
                _objective,
                flat_initial,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < -best_area:
                opt_points = result.x.reshape(N, 2)
                opt_points = np.array([_project_to_triangle(p) for p in opt_points])
                area = _min_triangle_area(opt_points)
                if area > best_area:
                    best_area = area
                    best_points = opt_points
        except Exception:
            pass
        
        # Also try Nelder-Mead for robustness
        if time.time() > deadline - 5.0:
            break
        try:
            result2 = minimize(
                _objective,
                flat_initial,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_points2 = result2.x.reshape(N, 2)
            opt_points2 = np.array([_project_to_triangle(p) for p in opt_points2])
            area2 = _min_triangle_area(opt_points2)
            if area2 > best_area:
                best_area = area2
                best_points = opt_points2
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: simple grid
        best_points = _make_initial_config(0)
    
    # Final check: ensure all points are in triangle
    best_points = np.array([_project_to_triangle(p) for p in best_points])
    
    return best_points
