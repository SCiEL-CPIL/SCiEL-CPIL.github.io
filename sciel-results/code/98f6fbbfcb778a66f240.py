import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees + rotation
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + a
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hex(px, py, hx, hy, hside, hangle_deg):
    """Check if point (px,py) is inside hexagon centered at (hx,hy) with side hside and rotation hangle_deg."""
    # Transform point to hexagon's local frame
    dx = px - hx
    dy = py - hy
    a = np.radians(hangle_deg)
    # Rotate point by -a
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Regular hexagon with flat-top orientation (vertices at 0,60,...)
    # In local frame, hexagon is defined by |x| <= hside, |y| <= hside*sqrt(3)/2, etc.
    # Actually for a regular hexagon with vertices at angles 0,60,...,300:
    # The hexagon can be described as: for each of the 6 edges, the point must be on the correct side
    # Edge normals point outward. For a hexagon with vertices at angle k*60 deg,
    # the edge between vertex k and k+1 has outward normal at angle (k+0.5)*60 deg
    # Distance from center to each edge = hside * sqrt(3)/2
    # Let's use the half-plane representation
    d = hside * np.sqrt(3) / 2.0  # apothem
    for k in range(6):
        theta = (k + 0.5) * np.pi / 3.0  # normal direction
        nx, ny = np.cos(theta), np.sin(theta)
        # Point must satisfy nx*lx + ny*ly <= d
        if nx * lx + ny * ly > d:
            return False
    return True

def hex_inside_hex(inner_cx, inner_cy, inner_side, inner_angle,
                    outer_cx, outer_cy, outer_side, outer_angle):
    """Check if inner hex is entirely inside outer hex by checking all inner vertices."""
    verts = hex_vertices(inner_cx, inner_cy, inner_side, inner_angle)
    for vx, vy in verts:
        if not point_in_hex(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def hex_overlap_dist(c1x, c1y, c1a, c2x, c2y, c2a):
    """Return negative distance if hexagons overlap (0 = touching, >0 = separated).
    We use a simple approximation: check if centers are closer than 2*apothem in some direction.
    For exactness we'd need polygon intersection, but for optimization we can use a proxy.
    Actually, let's compute the minimum distance between the two hexagons' boundaries.
    If min distance < 0, they overlap.
    """
    v1 = hex_vertices(c1x, c1y, 1.0, c1a)
    v2 = hex_vertices(c2x, c2y, 1.0, c2a)
    min_dist = float('inf')
    for i in range(6):
        p1 = v1[i]
        for j in range(6):
            p2 = v2[j]
            d = np.linalg.norm(p1 - p2)
            if d < min_dist:
                min_dist = d
    # Also check if one vertex is inside the other hex
    # For simplicity, if min vertex-vertex distance is small, they likely overlap
    # Better: check if any vertex of v1 is inside hex2 or vice versa
    for vx, vy in v1:
        if point_in_hex(vx, vy, c2x, c2y, 1.0, c2a):
            return -0.1  # definitely overlapping
    for vx, vy in v2:
        if point_in_hex(vx, vy, c1x, c1y, 1.0, c1a):
            return -0.1
    return min_dist

def objective(params):
    """Minimize outer side length. params = [x1,y1,a1, ..., x12,y12,a12, ox,oy,os,oa]
    Total: 12*3 + 4 = 40 params
    """
    elapsed = time.time() - START_TIME
    if elapsed > DEADLINE:
        return 1e6
    
    inner_params = params[:36].reshape(12, 3)
    ox, oy, os, oa = params[36:40]
    
    # Penalty for inner hexes not being inside outer
    penalty = 0.0
    for i in range(12):
        ix, iy, ia = inner_params[i]
        verts = hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in verts:
            if not point_in_hex(vx, vy, ox, oy, os, oa):
                # Compute how far outside
                # Simple penalty: distance outside
                penalty += 10.0
    
    # Penalty for overlapping inner hexes
    for i in range(12):
        for j in range(i+1, 12):
            ix1, iy1, ia1 = inner_params[i]
            ix2, iy2, ia2 = inner_params[j]
            d = hex_overlap_dist(ix1, iy1, ia1, ix2, iy2, ia2)
            if d < 0.05:  # overlapping or very close
                penalty += max(0, 0.05 - d) * 50.0
    
    return os + penalty

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial guess: hexagonal packing
    # Place hexagons in a hexagonal grid pattern
    # For 12 hexagons, a good initial arrangement:
    # Row 0: 3 hexes, Row 1: 4 hexes, Row 2: 3 hexes, Row 3: 2 hexes? 
    # Or: use a compact arrangement
    
    # Hexagonal grid spacing: for unit hexagons (side=1), center-to-center distance
    # for non-overlapping hexes in a hex grid is sqrt(3) (for flat-top) or 2*sqrt(3)/2 = sqrt(3)
    # Actually for regular hexagons of side 1, the distance between centers of adjacent
    # hexagons in a honeycomb is sqrt(3) * side = sqrt(3)
    
    s = np.sqrt(3)  # center-to-center distance for adjacent hexes
    
    # Try a few initial configurations
    best_result = None
    best_score = float('inf')
    
    # Configuration 1: 3-4-3-2 arrangement
    configs = []
    
    # Config: hexagonal cluster
    # Positions in hex grid coordinates (q, r) -> cartesian
    hex_positions = []
    # Center + 6 around + 5 more
    # Let's just place them in a grid
    # For a compact arrangement, use axial coordinates
    # Hex at (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # That's 7. Need 5 more: (2,-1), (-2,1), (1,1), (-1,-1), (2,0)?
    
    # Axial to cartesian for flat-top hexagons:
    # x = s * (q + r/2)
    # y = s * (sqrt(3)/2) * r
    # Wait, let me be more careful.
    
    # For a regular hexagon with side 1, flat-top orientation:
    # The center-to-center distance for adjacent hexes:
    # Horizontal neighbors: distance = sqrt(3) * 1 = sqrt(3)
    # Diagonal neighbors: also sqrt(3)
    
    # Axial coordinate (q, r):
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # 12 hex positions in a compact hexagonal shape
    # A hex number of order 2 has 1+6=7, order 3 has 19.
    # For 12, let's use: center(0,0), ring1: 6, ring2 partial: 5
    pos_axial = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
                 (2,-1), (-2,1), (1,1), (-1,-1), (2,0)]
    
    for q, r in pos_axial:
        x, y = axial_to_cartesian(q, r)
        hex_positions.append((x, y))
    
    # Initial outer hex: large enough
    max_r = max(np.hypot(x, y) for x, y in hex_positions)
    initial_outer_side = max_r + 2.0
    
    # Build initial params
    init_params = []
    for x, y in hex_positions:
        init_params.extend([x, y, 0.0])
    init_params.extend([0.0, 0.0, initial_outer_side, 0.0])
    init_params = np.array(init_params)
    
    # Run Nelder-Mead
    try:
        result = minimize(objective, init_params, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4,
                                 'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
        
        # Try another initial config: all in a row
        init_params2 = []
        for k in range(12):
            init_params2.extend([k * np.sqrt(3), 0.0, 0.0])
        init_params2.extend([6 * np.sqrt(3), 0.0, 15.0, 0.0])
        init_params2 = np.array(init_params2)
        
        if time.time() - START_TIME < DEADLINE - 30:
            result2 = minimize(objective, init_params2, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4,
                                     'adaptive': True})
            if result2.fun < best_score:
                best_score = result2.fun
                best_result = result2.x
        
        # Try with rotated outer hex
        if time.time() - START_TIME < DEADLINE - 30:
            init_params3 = []
            for x, y in hex_positions:
                init_params3.extend([x, y, 0.0])
            init_params3.extend([0.0, 0.0, initial_outer_side, 30.0])
            init_params3 = np.array(init_params3)
            
            result3 = minimize(objective, init_params3, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4,
                                     'adaptive': True})
            if result3.fun < best_score:
                best_score = result3.fun
                best_result = result3.x
        
    except Exception:
        pass
    
    if best_result is None:
        # Fallback: return the initial config
        best_result = init_params
    
    # Extract results
    inner = []
    for i in range(12):
        x = float(best_result[i*3])
        y = float(best_result[i*3+1])
        a = float(best_result[i*3+2])
        inner.append([x, y, a])
    
    outer_cx = float(best_result[36])
    outer_cy = float(best_result[37])
    outer_side = float(best_result[38])
    outer_angle = float(best_result[39])
    
    # Ensure side is reasonable
    if outer_side < 1.0:
        outer_side = 1.0
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle

START_TIME = time.time()
