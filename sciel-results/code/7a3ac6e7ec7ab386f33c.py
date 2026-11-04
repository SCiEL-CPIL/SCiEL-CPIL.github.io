"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, angle_deg, scale=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Unit hexagon vertices (circumradius 1)
    base = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    # Rotate
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    verts = (R @ base.T).T * scale + np.array([cx, cy])
    return verts

def _polygon_area_signed(verts):
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def _point_in_polygon(px, py, verts):
    """Ray casting point-in-polygon test."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _seg_seg_intersect(p1, p2, p3, p4):
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

def _polygons_overlap(v1, v2):
    """Check if two convex polygons overlap using separating axis theorem."""
    # Get all edges (normals)
    for verts in [v1, v2]:
        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            edge = verts[j] - verts[i]
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto this normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return False  # Found separating axis
    return True

def _min_dist_between_polygons(v1, v2):
    """Approximate minimum distance between two convex polygons."""
    # If overlapping, return 0
    if _polygons_overlap(v1, v2):
        return 0.0
    # Compute min distance between all edge pairs
    min_d = float('inf')
    for i in range(6):
        j1 = (i + 1) % 6
        for j in range(6):
            j2 = (j + 1) % 6
            # Point-to-segment distances
            for pt in [v1[i], v1[j1]]:
                d = _pt_seg_dist(pt, v2[j], v2[j2])
                min_d = min(min_d, d)
            for pt in [v2[j], v2[j2]]:
                d = _pt_seg_dist(pt, v1[i], v1[j1])
                min_d = min(min_d, d)
    return min_d

def _pt_seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def _compute_cost(params, deadline=None):
    """
    params layout:
    [0..10]: inner hex centers x (11 values)
    [11..21]: inner hex centers y (11 values)
    [22..32]: inner hex angles in degrees (11 values)
    [33]: outer center x
    [34]: outer center y
    [35]: outer side length
    [36]: outer rotation angle in degrees
    Total: 37 params
    """
    # Decode
    inner_x = params[0:11]
    inner_y = params[11:22]
    inner_ang = params[22:33]
    outer_cx = params[33]
    outer_cy = params[34]
    outer_side = params[35]
    outer_rot = params[36]
    
    if outer_side < 0.5:
        outer_side = 0.5
    
    # Compute outer hexagon vertices
    outer_verts = _hex_vertices(outer_cx, outer_cy, outer_rot, scale=outer_side)
    
    cost = 0.0
    
    # Check containment: all inner hex vertices must be inside outer hex
    for i in range(11):
        v = _hex_vertices(inner_x[i], inner_y[i], inner_ang[i])
        for vx, vy in v:
            if not _point_in_polygon(vx, vy, outer_verts):
                # Penalty: distance outside
                # Find closest point on outer boundary
                min_d = float('inf')
                for j in range(6):
                    k = (j + 1) % 6
                    d = _pt_seg_dist(np.array([vx, vy]), outer_verts[j], outer_verts[k])
                    min_d = min(min_d, d)
                cost += 100.0 * min_d * min_d
    
    # Check non-overlap between inner hexagons
    for i in range(11):
        v1 = _hex_vertices(inner_x[i], inner_y[i], inner_ang[i])
        for j in range(i+1, 11):
            v2 = _hex_vertices(inner_x[j], inner_y[j], inner_ang[j])
            if _polygons_overlap(v1, v2):
                # Heavy penalty for overlap
                cost += 1000.0
            else:
                # Small penalty if too close (encourages separation)
                d = _min_dist_between_polygons(v1, v2)
                if d < 0.1:
                    cost += 10.0 * (0.1 - d) * (0.1 - d)
    
    # Regularization: keep outer side length small
    cost += 0.01 * outer_side * outer_side
    
    return cost

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    # Initial configuration: hexagonal cluster
    # Center hexagon at origin, 6 around it, 4 more
    # For non-rotated hexagons, spacing is 2*sqrt(3)
    s = 2.0 * np.sqrt(3)  # ~3.464
    
    # Hexagonal lattice positions for 11 hexagons
    # 1 center, 6 in first ring, 4 in second ring (partial)
    positions = [
        (0, 0),
        (s, 0), (s/2, s*np.sqrt(3)/2), (-s/2, s*np.sqrt(3)/2),
        (-s, 0), (-s/2, -s*np.sqrt(3)/2), (s/2, -s*np.sqrt(3)/2),
        # 4 more: extend the cluster
        (3*s/2, s*np.sqrt(3)/2), (3*s/2, -s*np.sqrt(3)/2),
        (s, s*np.sqrt(3)), (0, s*np.sqrt(3)),
    ]
    
    # Better: use a more compact arrangement
    # Try: 3x3 grid minus corners = 7, plus 4 more
    # Or: honeycomb pattern
    # Let's use a proper hexagonal cluster of 19 (3 rings) and pick 11 best
    # Actually, let's just start with a reasonable cluster
    
    inner_x0 = [p[0] for p in positions]
    inner_y0 = [p[1] for p in positions]
    inner_ang0 = [0.0] * 11
    
    # Estimate outer size needed
    max_r = max(np.sqrt(x**2 + y**2) for x, y in positions)
    outer_side0 = max_r + 1.0  # Add margin for hexagon size
    outer_cx0 = 0.0
    outer_cy0 = 0.0
    outer_rot0 = 0.0  # Pointy-top outer
    
    x0 = inner_x0 + inner_y0 + inner_ang0 + [outer_cx0, outer_cy0, outer_side0, outer_rot0]
    
    # Nelder-Mead optimization
    best_params = np.array(x0, dtype=float)
    best_cost = _compute_cost(best_params)
    
    # Try multiple random starts
    np.random.seed(42)
    n_starts = 5
    for trial in range(n_starts):
        if time.time() > DEADLINE - 30:
            break
        
        if trial == 0:
            x_try = np.array(x0, dtype=float)
        else:
            # Perturb initial
            x_try = np.array(x0, dtype=float)
            x_try[:22] += np.random.uniform(-0.5, 0.5, 22)
            x_try[22:33] += np.random.uniform(-30, 30, 11)
            x_try[33:35] += np.random.uniform(-1, 1, 2)
            x_try[35] = max(2.0, x_try[35] * np.random.uniform(0.8, 1.2))
            x_try[36] = np.random.uniform(-30, 30)
        
        try:
            result = minimize(_compute_cost, x_try, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Final polish with tighter tolerances
    if time.time() < DEADLINE - 10:
        try:
            result = minimize(_compute_cost, best_params, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract results
    inner_x = best_params[0:11]
    inner_y = best_params[11:22]
    inner_ang = best_params[22:33]
    outer_cx = best_params[33]
    outer_cy = best_params[34]
    outer_side = best_params[35]
    outer_rot = best_params[36]
    
    # Binary search for minimum outer side that still contains all hexagons
    # First, verify current solution is valid
    def is_valid(side_len, rot=outer_rot):
        outer_verts = _hex_vertices(outer_cx, outer_cy, rot, scale=side_len)
        for i in range(11):
            v = _hex_vertices(inner_x[i], inner_y[i], inner_ang[i])
            for vx, vy in v:
                if not _point_in_polygon(vx, vy, outer_verts):
                    return False
        # Check non-overlap
        for i in range(11):
            v1 = _hex_vertices(inner_x[i], inner_y[i], inner_ang[i])
            for j in range(i+1, 11):
                v2 = _hex_vertices(inner_x[j], inner_y[j], inner_ang[j])
                if _polygons_overlap(v1, v2):
                    return False
        return True
    
    # Binary search for minimal side
    lo, hi = 1.0, outer_side + 5.0
    if not is_valid(hi):
        hi = outer_side + 10.0
    
    if is_valid(lo):
        outer_side_final = lo
    else:
        for _ in range(40):
            if time.time() > DEADLINE - 5:
                break
            mid = (lo + hi) / 2.0
            if is_valid(mid):
                hi = mid
            else:
                lo = mid
        outer_side_final = hi
    
    # Build return
    inner = [[float(inner_x[i]), float(inner_y[i]), float(inner_ang[i])] for i in range(11)]
    return inner, (float(outer_cx), float(outer_cy)), float(outer_side_final), float(outer_rot)
# EVOLVE-BLOCK-END
