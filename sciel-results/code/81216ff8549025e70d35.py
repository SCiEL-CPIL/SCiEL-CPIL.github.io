import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def point_in_hexagon(px, py, hx, hy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (hx,hy)."""
    # Transform to hexagon's local frame
    angle = np.radians(angle_deg)
    dx = px - hx
    dy = py - hy
    # Rotate point by -angle
    ca, sa = np.cos(-angle), np.sin(-angle)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # In local frame, hexagon with side s has vertices at angles 0,60,...,300
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        a = i * np.pi / 3
        nx = np.cos(a)
        ny = np.sin(a)
        dist = lx * nx + ly * ny
        if dist > apothem + 1e-9:
            return False
    return True

def all_points_in_hexagon(points, hx, hy, side, angle_deg):
    """Check if all points are inside the hexagon."""
    for px, py in points:
        if not point_in_hexagon(px, py, hx, hy, side, angle_deg):
            return False
    return True

def hexagon_contains_hexagon(inner_verts, outer_hx, outer_hy, outer_side, outer_angle):
    """Check if inner hexagon (given by vertices) is inside outer hexagon."""
    return all_points_in_hexagon(inner_verts, outer_hx, outer_hy, outer_side, outer_angle)

def check_no_overlap(inner_positions, inner_angles, outer_side):
    """Check that no two inner hexagons overlap. They should be disjoint."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = inner_positions[i]
            xj, yj = inner_positions[j]
            # Two unit hexagons overlap if their centers are too close
            # The minimum distance between centers of non-overlapping unit hexagons
            # depends on relative rotation. Conservative: use inradius*2 = sqrt(3)
            # But with rotation, the minimum separation can be less.
            # For safety, check actual polygon intersection
            vi = hex_vertices(xi, yi, 1.0, inner_angles[i])
            vj = hex_vertices(xj, yj, 1.0, inner_angles[j])
            # Simple check: if any vertex of one is inside the other, they overlap
            if all_points_in_hexagon(vi, xj, yj, 1.0, inner_angles[j]):
                return False
            if all_points_in_hexagon(vj, xi, yi, 1.0, inner_angles[i]):
                return False
            # Also check edge intersections for robustness
            if polygons_intersect(vi, vj):
                return False
    return True

def polygons_intersect(v1, v2):
    """Check if two convex polygons intersect."""
    n1 = len(v1)
    n2 = len(v2)
    # Check if any vertex of v1 is inside v2
    for i in range(n1):
        if point_in_polygon(v1[i], v2):
            return True
    # Check if any vertex of v2 is inside v1
    for i in range(n2):
        if point_in_polygon(v2[i], v1):
            return True
    # Check edge intersections
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(v1[i], v1[(i+1)%n1], v2[j], v2[(j+1)%n2]):
                return True
    return False

def point_in_polygon(p, poly):
    """Ray casting algorithm for point in polygon."""
    px, py = p
    n = len(poly)
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1)%n]
        if ((y1 > py) != (y2 > py)) and (px < (x2-x1)*(py-y1)/(y2-y1) + x1):
            inside = not inside
    return inside

def segments_intersect(p1, p2, p3, p4):
    """Check if line segments p1-p2 and p3-p4 intersect."""
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

def objective(params):
    """Objective: outer hexagon side length. params = [ox, oy, oangle, oside, then for each inner: ix, iy, iangle]"""
    n_inner = N
    ox, oy, oangle, oside = params[0], params[1], params[2], params[3]
    inner_pos = []
    inner_angles = []
    for i in range(n_inner):
        base = 4 + i*3
        ix, iy, iangle = params[base], params[base+1], params[base+2]
        inner_pos.append((ix, iy))
        inner_angles.append(iangle)
    
    # Check all inner hexagons are inside outer
    for i in range(n_inner):
        vi = hex_vertices(inner_pos[i][0], inner_pos[i][1], 1.0, inner_angles[i])
        if not all_points_in_hexagon(vi, ox, oy, oside, oangle):
            return 1e6 + oside
    
    # Check no overlap
    if not check_no_overlap(inner_pos, inner_angles, oside):
        return 1e6 + oside
    
    return oside

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial hexagonal packing: 11 hexagons in a hex pattern
    # Axial coordinates for hex grid: (q, r) where q+r+s=0
    # For 11 hexagons: center (0,0), 6 at distance 1, 6 at distance 2 (corners)
    # Convert axial to pixel: x = q + r/2, y = r * sqrt(3)/2
    # Then scale by sqrt(3) for unit hexagons
    
    sqrt3 = np.sqrt(3)
    # Axial coordinates for hex pattern of radius 1 (11 hexagons)
    axial_coords = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            s = -q - r
            if abs(s) <= 1 and abs(q) <= 1 and abs(r) <= 1:
                axial_coords.append((q, r))
    
    # That gives 7. Need 11, so use radius 2 corners
    # Actually for 11: 1 + 6 + 6 (the 6 at distance 2 in corner directions)
    # Let me just use the standard hex number: H(n) = 3n(n-1)+1
    # H(1)=1, H(2)=7, H(3)=19. So 11 is between H(2) and H(3).
    # Use 7 from radius 1 + 4 from radius 2 corners
    
    # Simpler: just place them in a hex pattern
    # Center + 6 neighbors + 4 more
    positions = []
    angles = []
    
    # Center
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # 6 neighbors at distance sqrt(3)
    for i in range(6):
        a = i * np.pi / 3
        x = sqrt3 * np.cos(a)
        y = sqrt3 * np.sin(a)
        positions.append((x, y))
        angles.append(0.0)
    
    # 4 more at distance 2*sqrt(3) in alternating directions
    for i in [0, 1, 3, 4]:
        a = i * np.pi / 3
        x = 2*sqrt3 * np.cos(a)
        y = 2*sqrt3 * np.sin(a)
        positions.append((x, y))
        angles.append(0.0)
    
    # Build initial params
    ox, oy, oangle, oside = 0.0, 0.0, 0.0, 4.5
    params = [ox, oy, oangle, oside]
    for (px, py), ang in zip(positions, angles):
        params.extend([px, py, ang])
    
    best_params = params[:]
    best_obj = objective(params)
    
    # Local optimization
    try:
        result = minimize(objective, params, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() < deadline:
            if result.fun < best_obj:
                best_params = result.x
                best_obj = result.fun
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        # Return best found
        ox, oy, oangle, oside = best_params[0], best_params[1], best_params[2], best_params[3]
        inner = []
        for i in range(N):
            base = 4 + i*3
            ix, iy, iangle = best_params[base], best_params[base+1], best_params[base+2]
            inner.append([ix, iy, np.degrees(iangle)])
        return inner, (ox, oy), oside, np.degrees(oangle)
    
    # Try a few more restarts if time permits
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        # Perturb initial solution
        noise = np.random.randn(len(params)) * 0.1
        p = best_params + noise
        try:
            result = minimize(objective, p, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_obj:
                best_params = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    ox, oy, oangle, oside = best_params[0], best_params[1], best_params[2], best_params[3]
    inner = []
    for i in range(N):
        base = 4 + i*3
        ix, iy, iangle = best_params[base], best_params[base+1], best_params[base+2]
        inner.append([ix, iy, np.degrees(iangle)])
    return inner, (ox, oy), oside, np.degrees(oangle)
