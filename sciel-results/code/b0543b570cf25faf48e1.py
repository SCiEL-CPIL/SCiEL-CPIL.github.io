"""Packing 12 unit hexagons in a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def point_in_polygon(x, y, polygon):
    """Ray casting algorithm for point in polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    
    # Proper intersection
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def polygons_disjoint_hex(hex1_verts, hex2_verts):
    """Check if two hexagons are disjoint (no interior overlap)."""
    for v in hex1_verts:
        if point_in_polygon(v[0], v[1], hex2_verts):
            return False
    for v in hex2_verts:
        if point_in_polygon(v[0], v[1], hex1_verts):
            return False
    for i in range(6):
        p1 = hex1_verts[i]
        p2 = hex1_verts[(i+1)%6]
        for j in range(6):
            p3 = hex2_verts[j]
            p4 = hex2_verts[(j+1)%6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    return True

def compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons fit.
    """
    outer_cx, outer_cy = outer_center
    a_outer = np.radians(outer_angle)
    apothem_factor = np.sqrt(3) / 2
    max_s = 0.0
    
    for (ix, iy), ia in zip(inner_positions, inner_angles):
        inner_verts = hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in inner_verts:
            dx = vx - outer_cx
            dy = vy - outer_cy
            for i in range(6):
                normal_angle = a_outer + (i + 0.5) * np.pi / 3
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                dist = dx * nx + dy * ny
                if dist > 0:
                    s_needed = dist / apothem_factor
                    if s_needed > max_s:
                        max_s = s_needed
    return max_s

def evaluate_configuration(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Evaluate a configuration. Returns (outer_side, is_valid).
    """
    n = len(inner_positions)
    hex_verts_list = []
    for i in range(n):
        hex_verts_list.append(hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i]))
    
    for i in range(n):
        for j in range(i+1, n):
            if not polygons_disjoint_hex(hex_verts_list[i], hex_verts_list[j]):
                return (float('inf'), False)
    
    outer_side = compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle)
    return (outer_side, True)

def generate_12_hex_cluster():
    """
    Generate a valid hexagonal lattice configuration of 12 unit hexagons.
    Uses a 1+6+5 configuration (center, ring 1, 5 from ring 2).
    """
    positions = []
    angles = [0.0] * 12
    
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1 (6 hexes)
    for i in range(6):
        theta = i * np.pi / 3
        x = 2.0 * np.cos(theta)
        y = 2.0 * np.sin(theta)
        positions.append((x, y))
        
    # Ring 2 (select 5 edge hexes to form a compact shape)
    # The 12 hexes in ring 2 are at:
    # Corners: (4,0), (2, 2sqrt3), etc.
    # Edges: (3, sqrt3), (3, -sqrt3), etc.
    # We pick the 5 "edge" hexes to keep the shape closer to a hexagon.
    # Specifically, we skip one to make it 12 total.
    # Let's pick the ones at angles 30, 90, 150, 210, 270 (relative to x-axis)
    # These correspond to the "edge" positions.
    edge_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2]
    for theta in edge_angles:
        # Distance for edge hexes is sqrt(3^2 + (sqrt3)^2) = sqrt(12) = 2*sqrt(3)
        r = 2 * np.sqrt(3)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions.append((x, y))
        
    return positions, angles

def construct():
    best_side = float('inf')
    best_result = None
    
    # Generate the initial lattice configuration
    init_positions, init_angles = generate_12_hex_cluster()
    
    # Initial outer hexagon parameters
    # We can estimate a good initial side length and angle
    # The cluster is roughly hexagonal. Let's try aligning with the lattice.
    init_outer_center = (0.0, 0.0)
    init_outer_angle = 0.0 # Flat top? Or pointy?
    # The lattice has flat tops (vertices at 0, 60...).
    # So outer hexagon should also have flat tops (angle 0).
    
    # Evaluate initial configuration
    side, valid = evaluate_configuration(init_positions, init_angles, init_outer_center, init_outer_angle)
    if valid:
        best_side = side
        best_result = (init_positions, init_angles, init_outer_center, init_outer_angle, side)
    
    # Optimize the configuration
    # We optimize positions and angles of inner hexagons, and outer hexagon params.
    # To ensure validity, we use a penalty for overlaps.
    n = len(init_positions)
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e10
            
        positions = []
        for i in range(n):
            positions.append((params[2*i], params[2*i+1]))
        angles = params[2*n : 2*n+n]
        outer_center = (params[2*n+n], params[2*n+n+1])
        outer_angle = params[2*n+n+2]
        
        # Check validity and compute side
        side, valid = evaluate_configuration(positions, angles, outer_center, outer_angle)
        if not valid:
            # Penalty for overlap
            return 1e10
        return side
        
    x0 = []
    for pos in init_positions:
        x0.extend([pos[0], pos[1]])
    x0.extend(init_angles)
    x0.extend([init_outer_center[0], init_outer_center[1], init_outer_angle])
    
    # Use Nelder-Mead for local refinement
    # It's robust for small non-linear problems
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if result.fun < 1e9:
            pos_res = []
            for i in range(n):
                pos_res.append((result.x[2*i], result.x[2*i+1]))
            ang_res = result.x[2*n : 2*n+n]
            oc_res = (result.x[2*n+n], result.x[2*n+n+1])
            oa_res = result.x[2*n+n+2]
            
            if result.fun < best_side:
                best_side = result.fun
                best_result = (pos_res, ang_res, oc_res, oa_res, result.fun)
    except Exception:
        pass
        
    # Fallback if optimization failed or didn't improve
    if best_result is None:
        # Use the trivial arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions, angles, outer_center, outer_angle, side = best_result
    
    # Format output
    inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
    
    return inner, outer_center, side, outer_angle
