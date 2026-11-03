"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side length, and rotation."""
    cx, cy = center
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3.0
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return vertices

def _polygon_area(vertices):
    """Shoelace formula for polygon area."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def _point_in_polygon(px, py, vertices):
    """Ray casting algorithm."""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _polygons_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap (including touching)."""
    # Check if any vertex of one is inside the other
    for pt in v1:
        if _point_in_polygon(pt[0], pt[1], v2):
            return True
    for pt in v2:
        if _point_in_polygon(pt[0], pt[1], v1):
            return True
    
    # Check edge intersections
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        for j in range(n2):
            if _segments_intersect(v1[i], v1[(i+1)%n1], v2[j], v2[(j+1)%n2]):
                return True
    return False

def _min_outer_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length of a regular hexagon (with given center and rotation)
    that contains all inner hexagons.
    """
    ocx, ocy = outer_center
    oangle_rad = np.radians(outer_angle_deg)
    
    # The outer hexagon has 6 sides. For each side, we need to find the maximum
    # distance from the center to any vertex of the inner hexagons, projected onto
    # the normal of that side.
    
    # For a regular hexagon with side length s and center at origin, rotated by angle a,
    # the vertices are at distance s from center at angles a + k*pi/3.
    # The inradius (apothem) is s * sqrt(3)/2.
    # A point (x,y) is inside the hexagon if for each of the 6 side normals,
    # the projection is <= apothem.
    
    # Side normals for a hexagon rotated by angle a:
    # The sides are perpendicular to directions a + k*pi/3 + pi/6 (for k=0..5)
    # Actually, let's think differently.
    
    # A regular hexagon with side s, center at origin, rotation angle a:
    # Vertices at: s * (cos(a + k*pi/3), sin(a + k*pi/3)) for k=0..5
    # The hexagon is the intersection of 6 half-planes.
    # For each edge, the normal points outward.
    
    # Edge k connects vertex k to vertex (k+1)%6.
    # The outward normal for edge k is in direction a + (k+0.5)*pi/3.
    # The distance from center to each edge (apothem) is s * cos(pi/6) = s * sqrt(3)/2.
    
    # So for a point p to be inside: for all k, dot(p - center, n_k) <= apothem
    # where n_k = (cos(a + (k+0.5)*pi/3), sin(a + (k+0.5)*pi/3))
    
    max_ratio = 0.0
    for i in range(N):
        ix, iy, iangle = inner[i]
        iv = _hex_vertices((ix, iy), 1.0, iangle)
        for vx, vy in iv:
            dx, dy = vx - ocx, vy - ocy
            for k in range(6):
                nk = np.array([np.cos(oangle_rad + (k + 0.5) * np.pi / 3.0),
                               np.sin(oangle_rad + (k + 0.5) * np.pi / 3.0)])
                proj = dx * nk[0] + dy * nk[1]
                # apothem = s * sqrt(3)/2, so s >= proj / (sqrt(3)/2) = 2*proj/sqrt(3)
                if proj > 0:
                    ratio = proj / (np.sqrt(3) / 2.0)
                    if ratio > max_ratio:
                        max_ratio = ratio
    
    return max_ratio

def _is_valid(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon and don't overlap each other."""
    # Check inner hexagons are inside outer
    ocx, ocy = outer_center
    oangle_rad = np.radians(outer_angle_deg)
    apothem = outer_side * np.sqrt(3) / 2.0
    
    for i in range(N):
        ix, iy, iangle = inner[i]
        iv = _hex_vertices((ix, iy), 1.0, iangle)
        for vx, vy in iv:
            dx, dy = vx - ocx, vy - ocy
            for k in range(6):
                nk = np.array([np.cos(oangle_rad + (k + 0.5) * np.pi / 3.0),
                               np.sin(oangle_rad + (k + 0.5) * np.pi / 3.0)])
                proj = dx * nk[0] + dy * nk[1]
                if proj > apothem + eps:
                    return False
    
    # Check inner hexagons don't overlap each other
    for i in range(N):
        for j in range(i+1, N):
            vi = _hex_vertices((inner[i][0], inner[i][1]), 1.0, inner[i][2])
            vj = _hex_vertices((inner[j][0], inner[j][1]), 1.0, inner[j][2])
            if _polygons_overlap(vi, vj):
                return False
    
    return True

def _objective(x, inner_indices, fixed_positions):
    """
    Objective function for optimization.
    x contains variable parameters.
    """
    # This is a simplified version - we'll use a different approach
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    # Strategy: Use a hexagonal lattice arrangement and optimize positions
    # Start with a good initial arrangement
    
    # Hexagonal lattice positions for 12 hexagons
    # A hexagonal cluster of 12 hexagons can be arranged as:
    # Center + ring of 6 + partial ring of 5, or other configurations
    
    # Let's try several initial configurations and optimize
    
    def _hex_lattice_positions(n_hex):
        """Generate hexagonal lattice positions."""
        positions = [(0, 0)]
        # Ring 1: 6 positions
        for k in range(6):
            angle = k * np.pi / 3
            x = 2 * np.cos(angle)
            y = 2 * np.sin(angle)
            positions.append((x, y))
        # Ring 2: add positions until we have n_hex
        # Hexagonal lattice vectors
        v1 = np.array([2.0, 0.0])
        v2 = np.array([1.0, np.sqrt(3)])
        for a in range(-2, 3):
            for b in range(-2, 3):
                pos = a * v1 + b * v2
                x, y = pos[0], pos[1]
                # Check if already in positions
                found = False
                for px, py in positions:
                    if abs(px - x) < 0.1 and abs(py - y) < 0.1:
                        found = True
                        break
                if not found and len(positions) < n_hex:
                    positions.append((x, y))
        return positions[:n_hex]
    
    # Generate initial lattice positions
    lattice_pos = _hex_lattice_positions(N)
    
    # Try different rotations of the entire arrangement
    for outer_rot in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline:
            break
        
        # Create initial inner hexagons at lattice positions with same rotation
        inner_init = []
        for px, py in lattice_pos:
            inner_init.append([px, py, float(outer_rot)])
        
        # Compute initial outer side
        outer_side_init = _min_outer_side(inner_init, (0.0, 0.0), float(outer_rot))
        
        # Now optimize: minimize outer_side by adjusting inner positions
        # Parameters: for each inner hex, (dx, dy) offset from lattice position
        # and possibly rotation, but let's keep rotations fixed for simplicity
        
        def _opt_objective(params):
            inner = []
            for i in range(N):
                dx, dy = params[2*i], params[2*i+1]
                inner.append([lattice_pos[i][0] + dx, lattice_pos[i][1] + dy, float(outer_rot)])
            return _min_outer_side(inner, (0.0, 0.0), float(outer_rot))
        
        x0 = np.zeros(2 * N)
        
        try:
            result = minimize(_opt_objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < best_score:
                inner_opt = []
                for i in range(N):
                    dx, dy = result.x[2*i], result.x[2*i+1]
                    inner_opt.append([lattice_pos[i][0] + dx, lattice_pos[i][1] + dy, float(outer_rot)])
                
                outer_side_opt = _min_outer_side(inner_opt, (0.0, 0.0), float(outer_rot))
                
                # Verify validity
                if _is_valid(inner_opt, (0.0, 0.0), outer_side_opt, float(outer_rot)):
                    if outer_side_opt < best_score:
                        best_score = outer_side_opt
                        best_result = (inner_opt, (0.0, 0.0), outer_side_opt, float(outer_rot))
        except Exception:
            pass
    
    # Also try a simple linear arrangement as fallback
    if best_result is None:
        # Fallback: use the seed arrangement but compute proper outer side
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        outer_side = _min_outer_side(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), outer_side, 0.0)
        best_score = outer_side
    
    # Final polish: try to further optimize the best result found
    if best_result is not None and time.time() < deadline:
        inner_best, center_best, side_best, angle_best = best_result
        
        def _polish_objective(params):
            inner = []
            for i in range(N):
                dx, dy = params[2*i], params[2*i+1]
                inner.append([inner_best[i][0] + dx, inner_best[i][1] + dy, angle_best])
            return _min_outer_side(inner, center_best, angle_best)
        
        x0 = np.zeros(2 * N)
        try:
            result = minimize(_polish_objective, x0, method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-10})
            
            if result.fun < best_score:
                inner_polish = []
                for i in range(N):
                    dx, dy = result.x[2*i], result.x[2*i+1]
                    inner_polish.append([inner_best[i][0] + dx, inner_best[i][1] + dy, angle_best])
                
                outer_side_polish = _min_outer_side(inner_polish, center_best, angle_best)
                
                if _is_valid(inner_polish, center_best, outer_side_polish, angle_best):
                    best_score = outer_side_polish
                    best_result = (inner_polish, center_best, outer_side_polish, angle_best)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
