import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local coordinate system
    a = np.deg2rad(angle_deg)
    dx, dy = px - cx, py - cy
    # Rotate by -a to align hexagon with axes
    rx = dx * np.cos(-a) - dy * np.sin(-a)
    ry = dx * np.sin(-a) + dy * np.cos(-a)
    # For a regular hexagon with side s, the constraint is max of 3 absolute values
    # The hexagon in local coords has vertices at angles 0,60,...,300
    # The supporting lines are at distances: side*cos(30) for sides perpendicular to axes
    # Actually for a hexagon with vertices at angle 0, the flat sides are at 30, 90, 150, etc.
    # The apothem (distance from center to side midpoint) = side * cos(30) = side * sqrt(3)/2
    # The three pairs of parallel sides are at angles 30, 90, 150 (and their opposites)
    # In local coords after rotation by -a, check:
    # |x*cos(30) + y*sin(30)| <= apothem
    # |x*cos(90) + y*sin(90)| <= apothem  => |y| <= apothem
    # |x*cos(150) + y*sin(150)| <= apothem
    apothem = side * np.sqrt(3) / 2
    c30, s30 = np.cos(np.pi/6), np.sin(np.pi/6)
    c90, s90 = np.cos(np.pi/2), np.sin(np.pi/2)
    c150, s150 = np.cos(5*np.pi/6), np.sin(5*np.pi/6)
    d1 = abs(rx * c30 + ry * s30)
    d2 = abs(rx * c90 + ry * s90)
    d3 = abs(rx * c150 + ry * s150)
    return max(d1, d2, d3) <= apothem + 1e-12

def _hex_distance_to_boundary(cx, cy, angle_deg, S, outer_angle_deg, ocx, ocy, side=1.0):
    """
    Minimum distance from hexagon to outer hexagon boundary.
    Positive means inside, negative means outside.
    Uses the fact that distance from a point to a convex polygon can be computed,
    but for a hexagon inside a hexagon, we can use the support function approach.
    
    For a regular hexagon of side S with center (ocx,ocy) and rotation outer_angle,
    the boundary consists of 6 sides. For each side, compute the signed distance
    from each vertex of the inner hexagon to that side, and take the minimum.
    """
    # Get outer hexagon vertices
    outer_verts = _hex_vertices(ocx, ocy, outer_angle_deg, S)
    inner_verts = _hex_vertices(cx, cy, angle_deg, side)
    
    min_dist = np.inf
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        # Edge vector
        ex, ey = v2[0] - v1[0], v2[1] - v1[1]
        edge_len = np.sqrt(ex**2 + ey**2)
        if edge_len < 1e-12:
            continue
        # Normal pointing inward (toward center)
        nx, ny = -ey / edge_len, ex / edge_len
        # Check if normal points toward center
        to_center_x = ocx - v1[0]
        to_center_y = ocy - v1[1]
        if nx * to_center_x + ny * to_center_y < 0:
            nx, ny = -nx, -ny
        # Signed distance from each inner vertex to this edge
        for vx, vy in inner_verts:
            d = (vx - v1[0]) * nx + (vy - v1[1]) * ny
            if d < min_dist:
                min_dist = d
    return min_dist

def _separation(i, j, centers, angles):
    """
    Minimum distance between two unit hexagons.
    Positive means separated, negative means overlapping.
    For two regular hexagons of side 1, the minimum center distance for non-overlap
    depends on relative rotation. We use a conservative bound:
    The maximum "radius" of a hexagon is 1.0 (center to vertex).
    So if center distance < 2.0, they might overlap.
    For tighter check, we use the actual geometry.
    """
    cx1, cy1 = centers[i]
    cx2, cy2 = centers[j]
    a1 = angles[i]
    a2 = angles[j]
    
    # Quick rejection: if centers are far apart
    dx, dy = cx2 - cx1, cy2 - cy1
    dist = np.sqrt(dx**2 + dy**2)
    if dist >= 2.0 + 1e-9:
        return dist - 2.0  # Conservative: max radius is 1
    
    # For tighter check, sample the boundary
    # Get vertices of both hexagons
    verts1 = _hex_vertices(cx1, cy1, a1, 1.0)
    verts2 = _hex_vertices(cx2, cy2, a2, 1.0)
    
    # Check if any vertex of hex1 is inside hex2
    for vx, vy in verts1:
        if _point_in_hexagon(vx, vy, cx2, cy2, a2, 1.0):
            return -1.0  # Overlap detected
    # Check if any vertex of hex2 is inside hex1
    for vx, vy in verts2:
        if _point_in_hexagon(vx, vy, cx1, cy1, a1, 1.0):
            return -1.0  # Overlap detected
    
    # Check edge-edge intersections
    # For simplicity, if no vertex is inside and dist < 2, assume potential overlap
    # This is a conservative check
    if dist < np.sqrt(3) - 1e-9:
        return -1.0  # Definitely overlapping
    
    return dist - 2.0  # Conservative separation

def _compute_outer_side(centers, angles, outer_angle_deg):
    """
    Compute the minimum side length of outer hexagon (with given rotation)
    that contains all inner hexagons.
    """
    # For each inner hexagon, find the maximum extent in the direction of each outer side normal
    # The outer hexagon has 6 sides. For each side, the constraint is that all inner
    # hexagon vertices must be on the inner side of that edge.
    
    # We'll binary search on S
    lo = 0.0
    hi = 10.0
    
    def fits(S):
        outer_verts = _hex_vertices(0.0, 0.0, outer_angle_deg, S)
        for i in range(len(centers)):
            inner_verts = _hex_vertices(centers[i][0], centers[i][1], angles[i], 1.0)
            for j in range(6):
                v1 = outer_verts[j]
                v2 = outer_verts[(j + 1) % 6]
                ex, ey = v2[0] - v1[0], v2[1] - v1[1]
                edge_len = np.sqrt(ex**2 + ey**2)
                if edge_len < 1e-12:
                    continue
                nx, ny = -ey / edge_len, ex / edge_len
                # Check if normal points inward
                if nx * (0.0 - v1[0]) + ny * (0.0 - v1[1]) < 0:
                    nx, ny = -nx, -ny
                for vx, vy in inner_verts:
                    d = (vx - v1[0]) * nx + (vy - v1[1]) * ny
                    if d < -1e-9:
                        return False
        return True
    
    for _ in range(50):
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    
    return hi + 1e-9

def _objective(params, outer_angle_deg):
    """
    Objective function: outer side length + penalty for overlaps.
    params: 22 values = 11 * (x, y), angles fixed at 30 degrees for simplicity
    """
    centers = np.array(params).reshape(11, 2)
    angles = np.full(11, 30.0)
    
    # Check for overlaps
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            sep = _separation(i, j, centers, angles)
            if sep < 0:
                penalty += sep * sep * 1000
    
    if penalty > 1e-6:
        return 1000.0 + penalty  # Invalid
    
    side = _compute_outer_side(centers, angles, outer_angle_deg)
    return side

def construct():
    start_time = time.time()
    
    # Initial flower configuration
    centers = np.zeros((11, 2))
    angles = np.full(11, 30.0)
    
    # Ring 0
    centers[0] = [0.0, 0.0]
    # Ring 1
    for k in range(6):
        a = np.deg2rad(k * 60)
        r = np.sqrt(3)
        centers[1 + k] = [r * np.cos(a), r * np.sin(a)]
    # Ring 2
    for k in range(4):
        a = np.deg2rad(30 + k * 60)
        r = 3.0
        centers[7 + k] = [r * np.cos(a), r * np.sin(a)]
    
    # Try different outer angles
    best_side = np.inf
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_outer_angle = 30.0
    
    outer_angles_to_try = [0.0, 15.0, 30.0, 45.0, 60.0]
    
    for oa in outer_angles_to_try:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Optimize positions for this outer angle
        params0 = centers.flatten()
        
        def obj(p):
            return _objective(p, oa)
        
        try:
            res = minimize(obj, params0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if res.fun < best_side:
                best_side = res.fun
                best_centers = res.x.reshape(11, 2)
                best_angles = angles.copy()
                best_outer_angle = oa
        except Exception:
            pass
    
    # If optimization didn't improve, use the initial configuration
    if best_side == np.inf:
        best_side = _compute_outer_side(centers, angles, 30.0)
        best_centers = centers
        best_angles = angles
        best_outer_angle = 30.0
    
    # Build inner list
    inner = []
    for i in range(11):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
    
    outer_center = (0.0, 0.0)
    side = best_side + 1e-9
    outer_angle = best_outer_angle
    
    return inner, outer_center, side, outer_angle
