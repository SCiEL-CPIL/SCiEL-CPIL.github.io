import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    angles = a + np.pi/6 + np.arange(6) * np.pi/3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side, angle."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    rx = dx * np.cos(a) + dy * np.sin(a)
    ry = -dx * np.sin(a) + dy * np.cos(a)
    R = side
    apothem = R * np.sqrt(3) / 2
    for k in range(6):
        theta = np.radians(30 + k * 60)
        val = rx * np.cos(theta) + ry * np.sin(theta)
        if val > apothem + 1e-9:
            return False
    return True

def _hexagon_contains_hex(cx1, cy1, a1, cx2, cy2, a2, side_inner=1.0, side_outer=None, outer_angle=0.0):
    """Check if hexagon 2 (inner) is fully inside hexagon 1 (outer)."""
    vx, vy = _hex_vertices(cx2, cy2, side_inner, a2)
    for i in range(6):
        if not _point_in_hex(vx[i], vy[i], cx1, cy1, side_outer, outer_angle):
            return False
    return True

def _hexagons_disjoint(cx1, cy1, a1, cx2, cy2, a2, side=1.0):
    """Check if two unit hexagons are disjoint (non-overlapping)."""
    d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    if d > 2.0 * side + 1e-9:
        return True
    vx1, vy1 = _hex_vertices(cx1, cy1, side, a1)
    vx2, vy2 = _hex_vertices(cx2, cy2, side, a2)
    for i in range(6):
        if _point_in_hex(vx1[i], vy1[i], cx2, cy2, side, a2):
            return False
    for i in range(6):
        if _point_in_hex(vx2[i], vy2[i], cx1, cy1, side, a1):
            return False
    return True

def _compute_min_side(positions, angles, center=(0.0, 0.0), outer_angle=0.0):
    """Compute the minimum outer hexagon side length that contains all inner hexagons."""
    max_side = 0.0
    for i in range(len(positions)):
        cx, cy = positions[i][0], positions[i][1]
        a = angles[i]
        vx, vy = _hex_vertices(cx, cy, 1.0, a)
        a_rad = np.radians(outer_angle)
        for j in range(6):
            px, py = vx[j], vy[j]
            dx = px - center[0]
            dy = py - center[1]
            rx = dx * np.cos(a_rad) + dy * np.sin(a_rad)
            ry = -dx * np.sin(a_rad) + dy * np.cos(a_rad)
            for k in range(6):
                theta = np.radians(30 + k * 60)
                proj = rx * np.cos(theta) + ry * np.sin(theta)
                required_S = 2.0 * abs(proj) / np.sqrt(3)
                if required_S > max_side:
                    max_side = required_S
    return max_side

def _objective(params, center, outer_angle):
    """Objective: outer hexagon side length."""
    n = N
    positions = []
    angles = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        a = params[3*i+2]
        positions.append((cx, cy))
        angles.append(a)
    
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if not _hexagons_disjoint(positions[i][0], positions[i][1], angles[i], 
                                      positions[j][0], positions[j][1], angles[j]):
                d = np.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)
                penalty += max(0, 2.0 - d) ** 2 * 100
    
    side = _compute_min_side(positions, angles, center, outer_angle)
    return side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Define a fixed, valid configuration of 11 unit hexagons.
    # We use a hexagonal lattice packing: 1 center, 6 surrounding, 4 more.
    # This configuration is disjoint and compact.
    sqrt3 = np.sqrt(3)
    
    # Centers of the 11 hexagons in a hexagonal grid (distance between adjacent centers is sqrt(3))
    # Row 0: 3 hexagons (y=0)
    # Row 1: 3 hexagons (y=sqrt(3)/2, x offset sqrt(3)/2)
    # Row 2: 3 hexagons (y=sqrt(3), x offset 0)
    # Row 3: 2 hexagons (y=3*sqrt(3)/2, x offset sqrt(3)/2)
    # Total 11.
    
    positions = []
    # We will generate positions relative to the cluster center, then rotate the whole cluster.
    # Base positions (flat-top hexagons aligned with axes)
    base_pos = []
    
    # Center
    base_pos.append((0.0, 0.0))
    
    # Ring 1 (6 hexagons)
    for i in range(6):
        angle = i * np.pi / 3
        # Distance to adjacent center is sqrt(3)
        x = sqrt(3) * np.cos(angle)
        y = sqrt(3) * np.sin(angle)
        base_pos.append((x, y))
        
    # Ring 2 (We need 4 more to make 11). 
    # In a perfect hexagonal packing of 19, ring 2 has 12. We take 4 of them to keep it compact.
    # The vertices of the "hexagon" formed by the 7 centers are at distance 2*sqrt(3) from center? No.
    # The centers of ring 2 are further out.
    # Let's just pick 4 positions from the next layer that are symmetric.
    # Next layer centers are at distance 2*sqrt(3) * cos(pi/6)? No.
    # Let's use the lattice vectors.
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5) ... wait, for flat top, adjacent centers are at distance sqrt(3).
    # Let's stick to the simple grid coordinates derived earlier.
    
    # Let's re-define base_pos using the grid logic which is safer for disjointness.
    # Grid spacing: dx = sqrt(3), dy = 1.5 (for pointy top) or similar.
    # For flat-top hexagons of side 1:
    # Width = 2, Height = sqrt(3).
    # Horizontal distance between centers in same row = 2 * (sqrt(3)/2) = sqrt(3)? No.
    # If flat-top, the "width" is 2 (vertex to vertex). The "height" is sqrt(3).
    # Adjacent hexagons share an edge. The distance between centers is 2 * apothem = sqrt(3).
    
    # Let's use the 3-3-3-2 grid from the original code but ensure it's valid.
    # The original code's grid was:
    # x = (col - 1) * sqrt3
    # y = row * 1.5
    # This corresponds to a specific lattice. Let's verify disjointness for this specific set.
    
    grid_pos = []
    for row in range(4):
        for col in range(3 if row < 3 else 2):
            x = (col - 1) * sqrt3
            if row % 2 == 1:
                x += sqrt3 / 2
            y = row * 1.5
            grid_pos.append((x, y))
            
    # This gives 11 points.
    # Let's center this cluster at (0,0) to make rotation simpler.
    cx_mean = np.mean([p[0] for p in grid_pos])
    cy_mean = np.mean([p[1] for p in grid_pos])
    centered_pos = [(p[0] - cx_mean, p[1] - cy_mean) for p in grid_pos]
    
    # The inner hexagons are all axis-aligned (angle 0) in this local frame.
    # We will rotate the entire cluster by an angle 'cluster_angle'.
    # The outer hexagon is fixed at angle 0 for calculation, so we minimize over cluster_angle.
    
    # Function to calculate required outer side length for a given cluster rotation
    def calc_side(cluster_angle_deg):
        cluster_angle_rad = np.radians(cluster_angle_deg)
        cos_a = np.cos(cluster_angle_rad)
        sin_a = np.sin(cluster_angle_rad)
        
        max_required_side = 0.0
        
        # Outer hexagon normals are at 30, 90, 150, 210, 270, 330 degrees.
        # Outer hexagon is centered at (0,0) and axis-aligned.
        # Vertices of inner hexagons need to be checked.
        
        for cx, cy in centered_pos:
            # Rotate center
            rx_c = cx * cos_a - cy * sin_a
            ry_c = cx * sin_a + cy * cos_a
            
            # The inner hexagon is also rotated by cluster_angle.
            # Its vertices in the global frame (relative to its rotated center)
            # are just the standard vertices rotated by cluster_angle.
            # Since the inner hexagon is rigid, its shape relative to the cluster is fixed.
            # We can pre-calculate the vertices of a unit hexagon at (0,0) angle 0.
            # Then rotate them by cluster_angle and translate to (rx_c, ry_c).
            
            # Vertices of unit hexagon (flat-top, angle 0) centered at 0,0:
            # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
            # Note: My _hex_vertices uses angle 0 -> pi/6 offset? 
            # In _hex_vertices: angles = a + pi/6 + ...
            # If a=0, vertices are at 30, 90, 150... degrees.
            # This is a "pointy-top" hexagon if we consider x horizontal?
            # cos(30)=0.866, sin(30)=0.5.
            # Let's stick to the _hex_vertices definition to be consistent.
            
            vx_local, vy_local = _hex_vertices(0.0, 0.0, 1.0, 0.0)
            
            # Rotate these local vertices by cluster_angle
            vx_rot = vx_local * cos_a - vy_local * sin_a
            vy_rot = vx_local * sin_a + vy_local * cos_a
            
            # Translate to global position
            vx_global = vx_rot + rx_c
            vy_global = vy_rot + ry_c
            
            # Check these vertices against the outer hexagon (center 0,0, angle 0)
            # Outer hexagon constraints:
            # For k in 0..5, normal angle theta_k = 30 + k*60 degrees.
            # proj = x * cos(theta_k) + y * sin(theta_k) <= apothem_outer
            # apothem_outer = S * sqrt(3) / 2
            # So S >= 2 * max(proj) / sqrt(3)
            
            for k in range(6):
                theta = np.radians(30 + k * 60)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                
                # Max projection of any vertex of this inner hexagon
                # proj_i = vx_global[i] * cos_t + vy_global[i] * sin_t
                projs = vx_global * cos_t + vy_global * sin_t
                max_proj = np.max(np.abs(projs))
                
                required_S = 2.0 * max_proj / np.sqrt(3)
                if required_S > max_required_side:
                    max_required_side = required_S
                    
        return max_required_side

    # Golden Section Search for optimal cluster_angle in [0, 60]
    # The function is periodic with 60 degrees.
    phi = (np.sqrt(5) - 1) / 2
    a = 0.0
    b = 60.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    
    fc = calc_side(c)
    fd = calc_side(d)
    
    # Iterate for precision
    for _ in range(50):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = calc_side(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = calc_side(d)
            
        if time.time() > deadline - 10.0:
            break
            
    opt_angle = (a + b) / 2.0
    min_side = calc_side(opt_angle)
    
    # Construct the final inner list
    # The inner hexagons are at centered_pos, rotated by opt_angle.
    # Their individual angles are also opt_angle (since they were 0 in local frame).
    # Wait, the interface asks for (x, y, angle_degrees).
    # If I rotate the cluster, the centers move, and the hexagons themselves rotate.
    
    inner = []
    cos_a = np.cos(np.radians(opt_angle))
    sin_a = np.sin(np.radians(opt_angle))
    
    for cx, cy in centered_pos:
        # Rotate center
        rx_c = cx * cos_a - cy * sin_a
        ry_c = cx * sin_a + cy * cos_a
        inner.append([float(rx_c), float(ry_c), float(opt_angle)])
        
    # Outer hexagon is centered at (0,0), angle 0.
    # Side length is min_side.
    # Add tiny buffer to account for floating point errors in evaluator's containment check.
    side = min_side + 1e-6
    
    return inner, (0.0, 0.0), float(side), 0.0

