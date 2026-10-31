"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices pts[i], pts[j], pts[k]
                # Using cross product formula
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    # Sort points by angle around centroid
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    sorted_pts = pts[order]
    
    # Graham scan
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    hull = []
    for p in sorted_pts:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    
    # Compute area of convex hull
    n_hull = len(hull)
    if n_hull < 3:
        return 0.0
    area = 0.0
    for i in range(n_hull):
        j = (i + 1) % n_hull
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def _objective(params):
    """Objective function to minimize (negative of the ratio)."""
    a, b, c, d = params
    # Affine transformation matrix
    M = np.array([[a, b], [c, d]])
    
    # Generate base 13-point grid: 4x4 grid minus one corner, centered
    # 4x4 grid points: (0,0), (0,1), (0,2), (0,3), (1,0), ..., (3,3)
    # Remove one corner, say (3,3), to get 15 points? No, 4x4=16, remove 3 to get 13.
    # Let's use a 4x4 grid and remove 3 points. Or better, use a fixed set of 13 points.
    # Let's use a centered 4x4 grid minus the 3 corners that are farthest from center? 
    # Actually, let's just use a fixed 13-point configuration.
    # A common approach: take a 4x4 grid and remove 3 points to get 13.
    # Let's remove the three corners that form an L-shape, keeping the configuration symmetric-ish.
    
    # Base grid: 4x4 from -1.5 to 1.5 (centered)
    coords = np.array([-1.5, -0.5, 0.5, 1.5])
    grid = np.array([[x, y] for y in coords for x in coords])
    # Remove 3 points to get 13. Let's remove (1.5, 1.5), (1.5, -1.5), (-1.5, 1.5)
    # That leaves 13 points.
    mask = np.ones(16, dtype=bool)
    # Indices: row 0: y=1.5, x=-1.5,-0.5,0.5,1.5 -> indices 0,1,2,3
    # row 1: y=0.5 -> 4,5,6,7
    # row 2: y=-0.5 -> 8,9,10,11
    # row 3: y=-1.5 -> 12,13,14,15
    # Remove (1.5,1.5)=index 3, (1.5,-1.5)=index 15, (-1.5,1.5)=index 0
    mask[0] = False
    mask[3] = False
    mask[15] = False
    base_pts = grid[mask]  # 13 points
    
    # Apply affine transformation
    transformed = base_pts @ M.T
    
    min_area = _min_triangle_area(transformed)
    hull_area = _convex_hull_area(transformed)
    
    if hull_area <= 1e-12:
        return 1e10
    
    ratio = min_area / hull_area
    return -ratio


def construct():
    """Construct 13 points maximizing the minimum triangle area ratio."""
    # Initial guess: identity-like transformation
    x0 = np.array([1.0, 0.0, 0.0, 1.0])
    
    # Optimize
    result = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    a, b, c, d = result.x
    M = np.array([[a, b], [c, d]])
    
    # Base grid
    coords = np.array([-1.5, -0.5, 0.5, 1.5])
    grid = np.array([[x, y] for y in coords for x in coords])
    mask = np.ones(16, dtype=bool)
    mask[0] = False
    mask[3] = False
    mask[15] = False
    base_pts = grid[mask]
    
    transformed = base_pts @ M.T
    
    # Normalize to unit area by scaling
    hull_area = _convex_hull_area(transformed)
    if hull_area > 1e-12:
        scale = 1.0 / np.sqrt(hull_area)
        transformed = transformed * scale
    
    return transformed
# EVOLVE-BLOCK-END
