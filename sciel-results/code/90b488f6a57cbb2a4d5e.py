import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side s, rotation angle_deg."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    theta = np.radians(angle_deg)
    # Rotate point by -theta
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    # In local frame, hexagon has vertices at angles 0,60,120,180,240,300 at distance s
    # The hexagon is the intersection of 6 half-planes
    # For each edge, check if point is inside
    for k in range(6):
        # Edge normal direction: angle k*60 + 30 (perpendicular to edge)
        # Actually, let's use the support function approach
        # A regular hexagon with "radius" (vertex distance) = s
        # The width in direction phi is 2*s*cos(min(|phi - k*60|, 60-|phi-k*60|)) ... 
        # Simpler: the hexagon contains (lx,ly) if for all 6 edge normals, dot product <= s
        # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees (for flat-top with vertex at 0)
        # Wait, if vertices are at 0,60,120,..., then edges are between consecutive vertices
        # The outward normal of edge between vertex k and k+1 is at angle (k+0.5)*60 = k*60+30
        normal_angle = np.radians(30 + k * 60)
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        dist = lx * nx + ly * ny
        if dist > s + 1e-9:
            return False
    return True

def hexagon_contains_all(points, cx, cy, s, angle_deg):
    """Check if all points are inside the hexagon."""
    for px, py in points:
        if not point_in_hexagon(px, py, cx, cy, s, angle_deg):
            return False
    return True

def min_enclosing_hexagon_side(points, cx, cy, angle_deg):
    """Find minimum side length of hexagon with given center and angle that contains all points."""
    # Binary search on side length
    lo = 0.0
    hi = 100.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if hexagon_contains_all(points, cx, cy, mid, angle_deg):
            hi = mid
        else:
            lo = mid
    return hi

def objective(params, points):
    """Objective: side length of enclosing hexagon given (cx, cy, angle_deg)."""
    cx, cy, angle_deg = params
    return min_enclosing_hexagon_side(points, cx, cy, angle_deg)

def get_lattice_points(axial_coords, s=1.0):
    """Convert axial coordinates to Cartesian for flat-top hexagons with side s."""
    pts = []
    for q, r in axial_coords:
        x = s * 1.5 * q
        y = s * np.sqrt(3) * (r + q / 2.0)
        pts.append((x, y))
    return pts

def try_arrangement(axial_coords, time_budget):
    """Try to find best enclosing hexagon for a given lattice arrangement."""
    points = get_lattice_points(axial_coords)
    
    # Compute centroid of points
    cx0 = np.mean([p[0] for p in points])
    cy0 = np.mean([p[1] for p in points])
    
    # Try multiple initial angles
    best_side = float('inf')
    best_params = None
    
    start_time = time.time()
    
    # Coarse search over angle
    for angle_init in range(0, 180, 15):
        if time.time() - start_time > time_budget:
            break
        # Use differential evolution for (cx, cy, angle)
        bounds = [(cx0 - 5, cx0 + 5), (cy0 - 5, cy0 + 5), (angle_init - 15, angle_init + 15)]
        
        def obj(p):
            return objective(p, points)
        
        result = differential_evolution(obj, bounds, maxiter=50, popsize=15, tol=1e-4, seed=42, polish=True)
        
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
    
    return best_side, best_params

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Define several candidate lattice arrangements of 12 hexagons in axial coordinates
    # Axial coords: (q, r) for flat-top hex grid
    
    # Pattern 1: 4x3 rectangular-ish
    pattern1 = [(q, r) for q in range(-2, 2) for r in range(-1, 2)]
    # That's 4*3=12
    
    # Pattern 2: Hexagonal cluster - center + 6 + 5 more
    # Center (0,0), ring1: (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)
    # Add 5 more from ring2
    pattern2 = [(0,0), (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),
                (2,0),(2,-1),(1,-2),(-1,-1),(0,2)]
    
    # Pattern 3: 3-4-3-2 arrangement
    pattern3 = [(-1,0),(0,0),(1,0),(2,0),
                (-2,-1),(-1,-1),(0,-1),(1,-1),
                (-1,-2),(0,-2),(1,-2),
                (0,-3)]
    
    # Pattern 4: 2 rows of 6 with offset
    pattern4 = [(q, 0) for q in range(-3, 3)] + [(q, -1) for q in range(-2, 4)]
    
    # Pattern 5: Compact hexagonal shape
    # Use cube coordinates converted to axial
    # Let's try a diamond/hex shape
    pattern5 = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),
                (2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-1)]
    
    patterns = [pattern1, pattern2, pattern3, pattern4, pattern5]
    
    best_side = float('inf')
    best_result = None
    
    for i, pattern in enumerate(patterns):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        # Allocate time proportionally, with minimum 3s each
        alloc = min(remaining * 0.4, max(3.0, remaining / (len(patterns) - i)))
        
        side, params = try_arrangement(pattern, alloc)
        
        if side < best_side and params is not None:
            best_side = side
            cx, cy, angle = params
            points = get_lattice_points(pattern)
            # Construct inner hexagons
            inner = []
            for (px, py) in points:
                inner.append([px, py, 0.0])  # inner hexagons at 0 degrees
            best_result = (inner, (cx, cy), best_side, angle)
    
    if best_result is None:
        # Fallback: simple arrangement
        inner = [[1.5 * k, 0.0, 0.0] for k in range(-5, 7)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-END
