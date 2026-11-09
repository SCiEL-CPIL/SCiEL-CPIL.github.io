"""Place 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 12
START_TIME = time.time()
DEADLINE = 105.0  # safety margin before 120s limit

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given angle."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + k * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def outer_hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of the outer regular hexagon."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + k * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def point_in_convex_polygon(pt, polygon):
    """Check if point is inside or on boundary of a convex polygon (CCW)."""
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross < -1e-9:
            return False
    return True

def hex_inside_outer(inner_verts, outer_verts, tol=1e-7):
    """Check if all vertices of inner hex are inside outer hex."""
    for v in inner_verts:
        if not point_in_convex_polygon(v, outer_verts):
            return False
    return True

def hex_hex_overlap(v1, v2, tol=1e-7):
    """Check if two hexagons (vertex arrays) overlap or touch.
    Return True if they are disjoint (no overlap, no touching)."""
    # Check if any vertex of v1 is inside v2 or vice versa
    for pt in v1:
        if point_in_convex_polygon(pt, v2, tol):
            return False
    for pt in v2:
        if point_in_convex_polygon(pt, v1, tol):
            return False
    return True

def segments_intersect(p1, p2, p3, p4):
    """Check if line segments p1-p2 and p3-p4 intersect (including touching)."""
    def orient(a, b, c):
        val = (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])
        if val > 1e-9:
            return 1
        elif val < -1e-9:
            return -1
        return 0
    
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    
    if d1 != d2 and d3 != d4:
        return True
    if d1 == 0 and point_on_segment(p3, p4, p1):
        return True
    if d2 == 0 and point_on_segment(p3, p4, p2):
        return True
    if d3 == 0 and point_on_segment(p1, p2, p3):
        return True
    if d4 == 0 and point_on_segment(p1, p2, p4):
        return True
    return False

def point_on_segment(a, b, p):
    """Check if point p lies on segment a-b."""
    if not (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9):
        return False
    if not (min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9):
        return False
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return abs(cross) < 1e-9

def hex_disjoint(v1, v2, tol=1e-7):
    """Check if two hexagons are disjoint (no overlap, no touching)."""
    # First check vertex containment
    for pt in v1:
        if point_in_convex_polygon(pt, v2):
            return False
    for pt in v2:
        if point_in_convex_polygon(pt, v1):
            return False
    # Check edge intersections
    for i in range(6):
        a1 = v1[i]
        a2 = v1[(i + 1) % 6]
        for j in range(6):
            b1 = v2[j]
            b2 = v2[(j + 1) % 6]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True

def evaluate_config(positions, angles, outer_center, outer_side, outer_angle=0.0):
    """Check if configuration is valid. positions: list of (x,y), angles: list of deg."""
    outer_verts = outer_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts_list = []
    for i in range(N):
        iv = hex_vertices(positions[i][0], positions[i][1], angles[i])
        inner_verts_list.append(iv)
        # Check inside outer
        if not hex_inside_outer(iv, outer_verts):
            return False, None
    # Check pairwise disjoint
    for i in range(N):
        for j in range(i + 1, N):
            if not hex_disjoint(inner_verts_list[i], inner_verts_list[j]):
                return False, None
    return True, inner_verts_list

def objective(params, outer_center, outer_side, outer_angle=0.0):
    """Objective function for optimization. Returns 0 if valid, penalty if invalid."""
    positions = [(params[i * 2], params[i * 2 + 1]) for i in range(N)]
    angles = params[2 * N:2 * N + N]
    
    outer_verts = outer_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts_list = []
    
    # Check inside outer
    penalty = 0.0
    for i in range(N):
        iv = hex_vertices(positions[i][0], positions[i][1], angles[i])
        inner_verts_list.append(iv)
        for v in iv:
            if not point_in_convex_polygon(v, outer_verts):
                # Compute distance outside
                # Find closest point on outer boundary
                min_dist = 1e10
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k + 1) % 6]
                    # Project v onto segment a-b
                    ab = b - a
                    av = v - a
                    if np.dot(ab, ab) < 1e-12:
                        dist = np.linalg.norm(v - a)
                    else:
                        t = np.dot(av, ab) / np.dot(ab, ab)
                        t = max(0, min(1, t))
                        closest = a + t * ab
                        dist = np.linalg.norm(v - closest)
                    min_dist = min(min_dist, dist)
                penalty += min_dist ** 2
    
    # Check pairwise disjoint - use center distance as rough proxy + vertex check
    for i in range(N):
        for j in range(i + 1, N):
            if not hex_disjoint(inner_verts_list[i], inner_verts_list[j]):
                # Compute minimum separation
                penalty += 10.0  # heavy penalty for overlap
                break
    
    return penalty

def construct():
    global START_TIME, DEADLINE
    
    # Initial configuration: hexagonal packing
    # For N=12, use a hexagonal arrangement
    # Center + 6 around + 5 more? Let's use a structured initial guess.
    
    # Hexagonal lattice positions for 12 hexagons
    # Use a pattern: 3 in bottom row, 4 in middle, 3 in second middle, 2 in top?
    # Actually, let's use a hexagonal cluster: center, 6 neighbors, then 5 more
    
    # Better: use a hexagonal grid
    # Positions in hex grid coordinates, convert to Cartesian
    # For unit hexagons, the distance between centers in hex grid is 2 (touching)
    
    # Let's start with a hexagonal arrangement
    # Center at origin, 6 around at distance 2, then 6 more in a ring
    # That's 1 + 6 + 6 = 13, too many. Let's use 1 + 6 + 5 = 12
    
    hex_dist = 2.0  # distance between centers of touching unit hexagons
    
    # Initial positions
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # 6 neighbors
    for k in range(6):
        angle = k * math.pi / 3
        x = hex_dist * math.cos(angle)
        y = hex_dist * math.sin(angle)
        positions.append((x, y))
    # 5 more in second ring (skip one to make 12 total)
    for k in range(6):
        if k == 5:  # skip one
            continue
        angle = k * math.pi / 3
        x = 2 * hex_dist * math.cos(angle)
        y = 2 * hex_dist * math.sin(angle)
        positions.append((x, y))
    
    angles = [0.0] * N
    
    # Initial outer side length estimate
    # The bounding hexagon should be about 4 units for this arrangement
    outer_center = (0.0, 0.0)
    outer_side = 6.0
    outer_angle = 0.0
    
    # Validate initial config
    valid, _ = evaluate_config(positions, angles, outer_center, outer_side, outer_angle)
    if not valid:
        # Adjust outer side
        outer_side = 7.0
        valid, _ = evaluate_config(positions, angles, outer_center, outer_side, outer_angle)
    
    best_config = (positions, angles, outer_center, outer_side, outer_angle)
    best_side = outer_side
    
    # Binary search on outer side length
    # Find minimum outer side that allows a valid configuration
    
    def find_min_side(pos_init, ang_init, o_center, o_angle=0.0, lo=None, hi=None):
        """Binary search for minimum outer side length."""
        if lo is None:
            lo = 3.0
        if hi is None:
            hi = 10.0
        
        # First ensure hi is valid
        for trial_side in [hi, hi * 1.2, hi * 1.5]:
            valid, _ = evaluate_config(pos_init, ang_init, o_center, trial_side, o_angle)
            if valid:
                hi = trial_side
                break
        else:
            return hi  # couldn't find valid upper bound
        
        for _ in range(40):
            if time.time() > DEADLINE:
                return hi
            mid = (lo + hi) / 2
            valid, _ = evaluate_config(pos_init, ang_init, o_center, mid, o_angle)
            if valid:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-4:
                break
        return hi
    
    # Try to minimize outer side with local optimization
    # Use scipy minimize on positions and angles for a given outer side
    
    def opt_for_side(side, init_pos, init_ang, o_center, o_angle=0.0):
        """Optimize positions and angles for given outer side."""
        x0 = []
        for p in init_pos:
            x0.extend([p[0], p[1]])
        x0.extend(init_ang)
        
        bounds = []
        R = side * 1.5  # rough bound
        for _ in range(N):
            bounds.append((-R, R))
            bounds.append((-R, R))
        bounds.extend([(-180, 180)] * N)
        
        try:
            result = minimize(
                objective,
                x0,
                args=(o_center, side, o_angle),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            return result.x
        except:
            return None
    
    # Iterative improvement: reduce outer side and re-optimize
    current_side = outer_side
    current_pos = list(positions)
    current_ang = list(angles)
    
    for iteration in range(30):
        if time.time() > DEADLINE:
            break
        
        # Try to reduce outer side
        new_side = current_side * 0.95
        min_side = find_min_side(current_pos, current_ang, outer_center, outer_angle, 
                                  lo=new_side, hi=current_side)
        
        if min_side < current_side - 1e-3:
            # Found smaller side, now optimize positions for this side
            opt_result = opt_for_side(min_side, current_pos, current_ang, outer_center, outer_angle)
            if opt_result is not None:
                # Check if valid
                new_pos = [(opt_result[i * 2], opt_result[i * 2 + 1]) for i in range(N)]
                new_ang = list(opt_result[2 * N:2 * N + N])
                valid, _ = evaluate_config(new_pos, new_ang, outer_center, min_side, outer_angle)
                if valid:
                    current_side = min_side
                    current_pos = new_pos
                    current_ang = new_ang
                    best_side = min_side
                    best_config = (current_pos, current_ang, outer_center, current_side, outer_angle)
        
        # Also try direct optimization to reduce side
        # Use a different approach: optimize with penalty on outer side
    
    # Final answer
    pos, ang, oc, oside, oangle = best_config
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
    return inner, (oc[0], oc[1]), oside, oangle
