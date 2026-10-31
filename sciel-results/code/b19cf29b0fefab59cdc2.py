"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11
H = np.sqrt(3) / 2.0  # height of the triangle

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside_triangle(pts, eps=1e-9):
    """Check if all points are inside the unit equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Check y bounds
    if np.any(y < -eps) or np.any(y > H + eps):
        return False
    # Check x bounds
    if np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    # Check left edge: x >= y / sqrt(3)
    if np.any(x < y / np.sqrt(3) - eps):
        return False
    # Check right edge: x <= 1 - y / sqrt(3)
    if np.any(x > 1 - y / np.sqrt(3) + eps):
        return False
    return True

def _project_to_triangle(pts, margin=1e-5):
    """Project points onto the triangle to ensure validity, keeping them strictly inside."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Ensure the triangle region defined by margins is valid
    # Width at height y is 1 - 2*y/sqrt(3).
    # We need 1 - 2*y/sqrt(3) > 2*margin => y < sqrt(3)/2 * (1 - 2*margin)
    max_y = H * (1 - 2 * margin)
    
    # Clamp y to [margin, max_y]
    y = np.clip(y, margin, max_y)
    
    # Clamp x based on y
    x_min = y / np.sqrt(3) + margin
    x_max = 1 - y / np.sqrt(3) - margin
    
    # If margin is too large for some y, x_min might exceed x_max, but max_y prevents that.
    x = np.clip(x, x_min, x_max)
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _barycentric_to_cartesian(b):
    """Convert barycentric coordinates to Cartesian."""
    # b is shape (n, 3), sum to 1
    x = b[:, 1] + 0.5 * b[:, 2]
    y = H * b[:, 2]
    return np.column_stack([x, y])

def _cartesian_to_barycentric(pts):
    """Convert Cartesian to barycentric coordinates."""
    x = pts[:, 0]
    y = pts[:, 1]
    # b0 = 1 - x - y/H
    # b1 = x
    # b2 = y/H
    b2 = y / H
    b1 = x - 0.5 * b2  # Wait, this is wrong.
    # Let's re-derive.
    # P = b0*V0 + b1*V1 + b2*V2
    # V0=(0,0), V1=(1,0), V2=(0.5, H)
    # x = b1*1 + b2*0.5
    # y = b2*H
    # So b2 = y/H
    # b1 = x - 0.5*b2
    # b0 = 1 - b1 - b2
    b2 = y / H
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _generate_hexagonal_init():
    """Generate a hexagonal grid initialization."""
    # Center at (0.5, H/3) roughly
    # Scale to fit
    # Hex grid points:
    pts = []
    # Center
    pts.append([0.5, H/3])
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        r = 0.15
        pts.append([0.5 + r * np.cos(angle), H/3 + r * np.sin(angle)])
    # Ring 2
    for i in range(6):
        angle = i * np.pi / 3
        r = 0.3
        pts.append([0.5 + r * np.cos(angle), H/3 + r * np.sin(angle)])
    return np.array(pts[:11])

def _generate_random_init():
    """Generate a random initialization inside the triangle."""
    pts = []
    for _ in range(N):
        # Random point in triangle
        u = np.random.rand()
        v = np.random.rand()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = H * v
        pts.append([x, y])
    return np.array(pts)

def _generate_perturbed_init():
    """Generate a perturbed hexagonal initialization."""
    pts = _generate_hexagonal_init()
    pts += np.random.normal(0, 0.05, pts.shape)
    pts = _project_to_triangle(pts)
    return pts

def _refine_local(pts):
    """Refine points locally using Nelder-Mead."""
    def objective(x):
        p = x.reshape(N, 2)
        p = _project_to_triangle(p)
        if not _check_inside_triangle(p, eps=-1e-6):
            return 1000.0
        return -_min_triangle_area(p)
    
    x0 = pts.flatten()
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    return res.x.reshape(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Initial configurations
    inits = [_generate_hexagonal_init(), _generate_random_init(), _generate_perturbed_init()]
    
    best_pts = None
    best_area = 0
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        init = _project_to_triangle(init)
        if not _check_inside_triangle(init):
            continue
        area = _min_triangle_area(init)
        if area > best_area:
            best_area = area
            best_pts = init.copy()
    
    if best_pts is None:
        best_pts = _generate_hexagonal_init()
        best_pts = _project_to_triangle(best_pts)
    
    # Local refinement
    if time.time() < deadline - 5:
        try:
            refined = _refine_local(best_pts)
            refined = _project_to_triangle(refined)
            if _check_inside_triangle(refined):
                area = _min_triangle_area(refined)
                if area > best_area:
                    best_area = area
                    best_pts = refined
        except Exception:
            pass
    
    # Final safety projection to ensure strict interior
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts
