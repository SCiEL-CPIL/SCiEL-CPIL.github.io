import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull

def compute_score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute minimum triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
                    
    if min_area == 0:
        return 0.0
        
    return min_area / hull_area

def construct():
    N = 13
    
    # Candidate 1: Best subset of 4x4 grid
    grid_4x4 = np.array([[x, y] for x in range(4) for y in range(4)], dtype=float)
    best_score_4x4 = 0.0
    best_points_4x4 = grid_4x4[:13] # Fallback
    
    # There are 16 points. We remove 3.
    # Indices 0 to 15.
    for remove_idx in combinations(range(16), 3):
        remove_set = set(remove_idx)
        mask = np.array([i not in remove_set for i in range(16)])
        pts = grid_4x4[mask]
        score = compute_score(pts)
        if score > best_score_4x4:
            best_score_4x4 = score
            best_points_4x4 = pts
            
    # Candidate 2: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    poly_13 = np.column_stack([np.cos(angles), np.sin(angles)])
    score_poly = compute_score(poly_13)
    
    # Candidate 3: Hexagonal cluster (approx)
    # Center + 6 neighbors + 6 more
    hex_pts = []
    # Basis
    u = np.array([1.0, 0.0])
    v = np.array([0.5, np.sqrt(3)/2])
    
    # Generate hex lattice points within radius 2
    # coords in axial coordinates (q, r)
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check distance
            # Convert to cartesian
            x = u[0]*q + v[0]*r
            y = u[1]*q + v[1]*r
            if np.sqrt(x*x + y*y) <= 2.1:
                hex_pts.append([x, y])
                
    hex_pts = np.array(hex_pts)
    # We have more than 13 points?
    # Radius 0: 1
    # Radius 1: 6
    # Radius 2: 12
    # Total 19.
    # We need to pick 13.
    # This is a subset search problem again.
    # Given the small size, we can try a few heuristics or just skip if 4x4 is good.
    # Let's just use the 4x4 result.
    
    if score_poly > best_score_4x4:
        return poly_13
    else:
        return best_points_4x4

N = 13
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull

def _compute_score(points):
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    if min_area == 0:
        return 0.0
    return min_area / hull_area

def construct():
    N = 13
    grid_4x4 = np.array([[x, y] for x in range(4) for y in range(4)], dtype=float)
    best_score = 0.0
    best_points = grid_4x4[:13]
    
    for remove_idx in combinations(range(16), 3):
        remove_set = set(remove_idx)
        mask = np.array([i not in remove_set for i in range(16)])
        pts = grid_4x4[mask]
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            
    return best_points
# EVOLVE-BLOCK-END
