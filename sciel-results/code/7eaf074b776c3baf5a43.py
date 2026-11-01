import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(angle_deg, cx=0.0, cy=0.0):
    """Vertices of unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + np.cos(theta), cy + np.sin(theta)))
    return np.array(verts)

def _outer_hex_vertices(angle_deg, side, cx=0.0, cy=0.0):
    """Vertices of regular hexagon with given side length, centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _point_in_polygon(pt, poly):
    """Ray casting point-in-polygon test."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _segment_intersection(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (proper intersection)."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_overlap(hex1_verts, hex2_verts, tol=1e-9):
    """Check if two hexagons overlap (interior intersection)."""
    # Check if any vertex of one is inside the other
    for v in hex1_verts:
        if _point_in_polygon(v, hex2_verts):
            return True
    for v in hex2_verts:
        if _point_in_polygon(v, hex1_verts):
            return True
    # Check edge intersections
    for i in range(6):
        for j in range(6):
            p1 = hex1_verts[i]
            p2 = hex1_verts[(i+1) % 6]
            p3 = hex2_verts[j]
            p4 = hex2_verts[(j+1) % 6]
            if _segment_intersection(p1, p2, p3, p4):
                return True
    return False

def _min_side_for_layout(positions, angles_deg):
    """
    Given positions and angles of N unit hexagons, find minimum outer hexagon side
    that contains all of them. We optimize over outer center and rotation.
    """
    best_side = float('inf')
    
    # Try several outer rotations and centers
    for outer_rot in range(0, 30, 5):  # outer rotation
        for cx_try in [-2, -1, 0, 1, 2]:
            for cy_try in [-2, -1, 0, 1, 2]:
                # For a given outer hex, find minimum side that contains all inner hexes
                # Binary search on side length
                lo, hi = 0.0, 20.0
                for _ in range(30):
                    mid = (lo + hi) / 2
                    outer_verts = _outer_hex_vertices(outer_rot, mid, cx_try, cy_try)
                    all_inside = True
                    for k in range(N):
                        inner_verts = _hex_vertices(angles_deg[k], positions[k][0], positions[k][1])
                        # Check all vertices of inner hex are inside outer
                        for v in inner_verts:
                            if not _point_in_polygon(v, outer_verts):
                                all_inside = False
                                break
                        if not all_inside:
                            break
                    if all_inside:
                        hi = mid
                    else:
                        lo = mid
                if hi < best_side:
                    best_side = hi
    
    return best_side

def _objective(params):
    """
    Objective: minimize the required outer side length.
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    """
    positions = []
    angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        positions.append((x, y))
        angles.append(a)
    
    # Check overlaps (penalty)
    overlap_penalty = 0.0
    hex_verts_list = []
    for k in range(N):
        hv = _hex_vertices(angles[k], positions[k][0], positions[k][1])
        hex_verts_list.append(hv)
    
    for i in range(N):
        for j in range(i+1, N):
            if _hexagons_overlap(hex_verts_list[i], hex_verts_list[j]):
                overlap_penalty += 100.0
    
    if overlap_penalty > 0:
        return overlap_penalty
    
    side = _min_side_for_layout(positions, angles)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial layout: hexagonal cluster
    # Center hexagon at origin, 6 around it, 4 more
    # For regular hex packing, adjacent centers are distance 2*inradius = sqrt(3) apart
    # Actually for unit hexagons (side=1), the distance between centers of touching hexes
    # depends on orientation. For aligned hexes, center-to-center = 2*(sqrt(3)/2) = sqrt(3)
    # in the "flat" direction, or 2 in the "pointy" direction.
    
    # Let's use a simple grid-like arrangement first
    # Hexagonal close packing: rows offset
    sqrt3 = np.sqrt(3)
    
    # Start with a compact arrangement
    initial_positions = []
    initial_angles = []
    
    # 3-4-3-1 or similar pattern
    # Let's try: center + 6 ring + 4 more
    # Actually for 11, a good pattern might be a 3x3 grid with corners removed, or hexagonal
    
    # Simple: place in a hexagonal pattern
    # Row 0: 3 hexes
    # Row 1: 4 hexes  
    # Row 2: 3 hexes
    # Row 3: 1 hex
    # That's 11
    
    # For aligned hexagons (flat-top), horizontal spacing = sqrt(3), vertical = 1.5
    # Offset rows shift by sqrt(3)/2
    
    layout = [
        # (row, col) positions
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2),
        (3, 1),
    ]
    
    for row, col in layout:
        x = col * sqrt3 + (row % 2) * sqrt3/2
        y = row * 1.5
        initial_positions.append((x, y))
        initial_angles.append(0.0)
    
    # Build initial params
    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = initial_positions[i][0]
        init_params[3*i+1] = initial_positions[i][1]
        init_params[3*i+2] = initial_angles[i]
    
    best_params = init_params.copy()
    best_score = _objective(init_params)
    
    # Try Nelder-Mead from multiple starting points
    start_points = [init_params]
    
    # Add perturbed versions
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbed = init_params + rng.normal(0, 0.5, size=3*N)
        start_points.append(perturbed)
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best solution
    positions = []
    angles = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        positions.append((x, y))
        angles.append(a)
    
    # Find best outer hexagon
    best_side = float('inf')
    best_outer_center = (0.0, 0.0)
    best_outer_rot = 0.0
    
    for outer_rot in range(0, 30, 2):
        for cx_try in np.arange(-3, 4, 0.5):
            for cy_try in np.arange(-3, 4, 0.5):
                lo, hi = 0.0, 20.0
                for _ in range(25):
                    mid = (lo + hi) / 2
                    outer_verts = _outer_hex_vertices(outer_rot, mid, cx_try, cy_try)
                    all_inside = True
                    for k in range(N):
                        inner_verts = _hex_vertices(angles[k], positions[k][0], positions[k][1])
                        for v in inner_verts:
                            if not _point_in_polygon(v, outer_verts):
                                all_inside = False
                                break
                        if not all_inside:
                            break
                    if all_inside:
                        hi = mid
                    else:
                        lo = mid
                if hi < best_side:
                    best_side = hi
                    best_outer_center = (cx_try, cy_try)
                    best_outer_rot = outer_rot
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
    
    return inner, best_outer_center, best_side, best_outer_rot
# EVOLVE-BLOCK-END
