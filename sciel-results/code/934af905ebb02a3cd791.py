import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 12
DEADLINE = 120.0  # wall-clock seconds

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy).
    
    The hexagon has flat-top orientation when angle_deg=0.
    Vertices are at angles 0, 60, 120, 180, 240, 300 degrees from center,
    rotated by angle_deg.
    """
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    return np.column_stack([cx + side * np.cos(angles), 
                            cy + side * np.sin(angles)])

def _outer_hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of outer hexagon. Same convention as inner."""
    return _hexagon_vertices(cx, cy, side, angle_deg)

def _polygon_containment_check(poly1, poly2, eps=1e-9):
    """Check if polygon poly2 is entirely outside (or non-overlapping) poly1.
    
    For convex polygons, they are non-overlapping if:
    1. No vertex of poly2 is inside poly1, AND
    2. No vertex of poly1 is inside poly2, AND  
    3. No edges intersect.
    
    Actually, for packing we need to check that inner hexagons don't overlap
    each other AND stay inside the outer hexagon.
    """
    pass

def _point_in_convex_polygon(point, polygon):
    """Check if point is inside convex polygon (CCW or CW)."""
    n = len(polygon)
    sign1 = None
    for i in range(n):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % n]
        cross = (v2[0] - v1[0]) * (point[1] - v1[1]) - (v2[1] - v1[1]) * (point[0] - v1[0])
        if abs(cross) < 1e-12:
            continue
        if sign1 is None:
            sign1 = np.sign(cross)
        elif np.sign(cross) != sign1:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects line segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_overlap(hex1, hex2, eps=1e-8):
    """Check if two convex hexagons overlap."""
    n1 = len(hex1)
    n2 = len(hex2)
    
    # Check if any vertex of hex1 is inside hex2
    for v in hex1:
        if _point_in_convex_polygon(v, hex2):
            return True
    
    # Check if any vertex of hex2 is inside hex1
    for v in hex2:
        if _point_in_convex_polygon(v, hex1):
            return True
    
    # Check edge intersections
    for i in range(n1):
        p1 = hex1[i]
        p2 = hex1[(i + 1) % n1]
        for j in range(n2):
            p3 = hex2[j]
            p4 = hex2[(j + 1) % n2]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    
    return False

def _hexagon_inside_outer(hex_inner, hex_outer, eps=1e-8):
    """Check if inner hexagon is entirely inside outer hexagon."""
    # All vertices of inner must be inside or on outer
    for v in hex_inner:
        if not _point_in_convex_polygon(v, hex_outer):
            return False
    return True

def _min_distance_between_hexagons(hex1, hex2):
    """Compute minimum distance between two convex hexagons (for penalty)."""
    # Approximate by checking vertex-to-vertex and vertex-to-edge distances
    min_d = np.inf
    for v1 in hex1:
        for v2 in hex2:
            d = np.linalg.norm(v1 - v2)
            if d < min_d:
                min_d = d
        # Also check distance from v1 to each edge of hex2
        for i in range(6):
            a = hex2[i]
            b = hex2[(i + 1) % 6]
            ab = b - a
            t = np.dot(v1 - a, ab) / np.dot(ab, ab)
            t = max(0, min(1, t))
            closest = a + t * ab
            d = np.linalg.norm(v1 - closest)
            if d < min_d:
                min_d = d
    return min_d

def _min_distance_to_boundary(hex_inner, hex_outer):
    """Compute minimum distance from inner hexagon to outer hexagon boundary."""
    min_d = np.inf
    for v in hex_inner:
        for i in range(6):
            a = hex_outer[i]
            b = hex_outer[(i + 1) % 6]
            ab = b - a
            t = np.dot(v - a, ab) / np.dot(ab, ab)
            t = max(0, min(1, t))
            closest = a + t * ab
            d = np.linalg.norm(v - closest)
            if d < min_d:
                min_d = d
    return min_d

def _packing_penalty(centers, angles, outer_side, outer_center, outer_angle, 
                      weight_overlap=1000.0, weight_boundary=100.0):
    """
    Compute a penalty for invalid packing configurations.
    Returns (penalty, is_valid).
    """
    penalty = 0.0
    is_valid = True
    
    outer_hex = _outer_hexagon_vertices(outer_center[0], outer_center[1], 
                                         outer_side, outer_angle)
    
    inner_hexes = []
    for i in range(N):
        cx, cy = centers[i]
        ang = angles[i]
        hex_i = _hexagon_vertices(cx, cy, 1.0, ang)
        inner_hexes.append(hex_i)
        
        # Check if inside outer
        if not _hexagon_inside_outer(hex_i, outer_hex):
            # Distance to boundary
            d = _min_distance_to_boundary(hex_i, outer_hex)
            if d < 0:
                # Vertically outside - use vertex distance
                min_vdist = np.inf
                for v in hex_i:
                    for j in range(6):
                        a = outer_hex[j]
                        b = outer_hex[(j + 1) % 6]
                        ab = b - a
                        t = np.dot(v - a, ab) / np.dot(ab, ab)
                        t = max(0, min(1, t))
                        closest = a + t * ab
                        dist = np.linalg.norm(v - closest)
                        if dist < min_vdist:
                            min_vdist = dist
                penalty += weight_boundary * min_vdist ** 2
                is_valid = False
            else:
                penalty += weight_boundary * d ** 2
                is_valid = False
    
    # Check pairwise overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagons_overlap(inner_hexes[i], inner_hexes[j]):
                d = _min_distance_between_hexagons(inner_hexes[i], inner_hexes[j])
                penalty += weight_overlap * max(0, -d) ** 2
                is_valid = False
    
    return penalty, is_valid

def _compute_min_outer_side(centers, angles, outer_center, outer_angle):
    """
    Given inner hexagon positions, compute the minimum outer side length
    that contains all inner hexagons.
    """
    inner_hexes = []
    for i in range(N):
        cx, cy = centers[i]
        ang = angles[i]
        hex_i = _hexagon_vertices(cx, cy, 1.0, ang)
        inner_hexes.append(hex_i)
    
    # For a given outer hexagon orientation and center, find minimum side
    # such that all inner hexagons are contained.
    # This is a complex problem. Approximate by checking containment
    # at various side lengths.
    
    # Simple approach: find the maximum distance from outer center to any
    # inner hexagon vertex, then compute required side.
    
    # Actually, for a regular hexagon, the distance from center to a point
    # depends on the angle. Let's use a simpler heuristic:
    # The circumradius of the outer hexagon must be at least the max
    # distance from outer center to any inner vertex.
    
    max_dist = 0.0
    for hex_i in inner_hexes:
        for v in hex_i:
            d = np.linalg.norm(v - np.array(outer_center))
            if d > max_dist:
                max_dist = d
    
    # For a regular hexagon with side s, circumradius = s
    # So minimum side = max_dist
    # But this is only valid if the point is in the "circum" direction.
    # For a point at angle theta from center, the required circumradius
    # is max_dist / cos(delta) where delta is the angle offset.
    
    # More accurate: for a regular hexagon, the distance from center to 
    # boundary at angle theta (measured from a vertex) is:
    # r(theta) = s / (cos(theta mod 60) * ... )
    
    # Let's use a more robust approach: binary search on side length
    low = max_dist * 0.5
    high = max_dist * 2.0
    
    for _ in range(30):
        mid = (low + high) / 2
        outer_hex = _outer_hexagon_vertices(outer_center[0], outer_center[1], 
                                             mid, outer_angle)
        valid = True
        for hex_i in inner_hexes:
            if not _hexagon_inside_outer(hex_i, outer_hex):
                valid = False
                break
        if valid:
            high = mid
        else:
            low = mid
    
    return high

def _generate_initial_config(config_type):
    """Generate an initial configuration for inner hexagons."""
    if config_type == 0:
        # Honeycomb-like arrangement
        # 3 rows of 4, offset
        centers = []
        angles = []
        row_height = np.sqrt(3)  # distance between rows in honeycomb
        for row in range(3):
            for col in range(4):
                x = col * 2.0 + (row % 2) * 1.0
                y = row * row_height
                centers.append((x, y))
                angles.append(0.0)
        return np.array(centers), np.array(angles)
    
    elif config_type == 1:
        # Ring arrangement: 1 center + 5 around + 6 more
        centers = []
        angles = []
        # Center
        centers.append((0, 0))
        angles.append(0.0)
        # First ring: 6 at distance ~2
        for k in range(6):
            ang = np.deg2rad(k * 60)
            x = 2.0 * np.cos(ang)
            y = 2.0 * np.sin(ang)
            centers.append((x, y))
            angles.append(0.0)
        # Second partial ring: 5 more
        for k in range(5):
            ang = np.deg2rad(k * 72 + 36)
            x = 3.5 * np.cos(ang)
            y = 3.5 * np.sin(ang)
            centers.append((x, y))
            angles.append(0.0)
        return np.array(centers), np.array(angles)
    
    elif config_type == 2:
        # Grid with rotation
        centers = []
        angles = []
        for row in range(3):
            for col in range(4):
                x = col * 2.1
                y = row * 1.8
                centers.append((x, y))
                angles.append(30.0)
        return np.array(centers), np.array(angles)
    
    elif config_type == 3:
        # Striped arrangement
        centers = []
        angles = []
        for i in range(12):
            x = (i % 4) * 2.2
            y = (i // 4) * 2.0
            centers.append((x, y))
            angles.append(0.0)
        return np.array(centers), np.array(angles)
    
    else:
        # Random
        centers = np.random.uniform(-4, 4, (N, 2))
        angles = np.random.uniform(0, 60, N)
        return centers, angles

def _optimize_from_initial(centers0, angles0, outer_center, outer_angle, time_limit):
    """Optimize from an initial configuration using Nelder-Mead."""
    start_time = time.time()
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return 1e10
        
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        
        penalty, is_valid = _packing_penalty(centers, angles, 20.0, 
                                              outer_center, outer_angle)
        if not is_valid:
            return 1e6 + penalty
        
        # Compute actual minimum outer side
        side = _compute_min_outer_side(centers, angles, outer_center, outer_angle)
        return side + penalty * 0.01
    
    params0 = np.concatenate([centers0.flatten(), angles0])
    
    try:
        result = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4,
                                  'adaptive': True})
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return None
        
        if result.success:
            centers = result.x[:2*N].reshape(N, 2)
            angles = result.x[2*N:3*N]
            side = _compute_min_outer_side(centers, angles, outer_center, outer_angle)
            
            # Verify
            penalty, is_valid = _packing_penalty(centers, angles, side, 
                                                  outer_center, outer_angle)
            if is_valid:
                return (centers, angles, side)
            else:
                return None
        else:
            return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    best_result = None
    best_side = float('inf')
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    config_types = [0, 1, 2, 3, 4, 4, 4]
    
    for cfg_type in config_types:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        centers0, angles0 = _generate_initial_config(cfg_type)
        
        # Add some randomness to break symmetry
        if cfg_type != 4:
            centers0 = centers0 + np.random.uniform(-0.3, 0.3, centers0.shape)
            angles0 = angles0 + np.random.uniform(-5, 5, len(angles0))
        
        time_per_opt = min(15.0, deadline - (time.time() - start_time))
        if time_per_opt < 3.0:
            break
        
        result = _optimize_from_initial(centers0, angles0, outer_center, 
                                         outer_angle, time_per_opt)
        
        if result is not None:
            centers, angles, side = result
            if side < best_side:
                best_side = side
                best_result = (centers, angles, side)
    
    if best_result is None:
        # Fallback: return the seed-like arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    centers, angles, side = best_result
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i][0]), float(centers[i][1]), float(angles[i])])
    
    return inner, outer_center, float(side), outer_angle
