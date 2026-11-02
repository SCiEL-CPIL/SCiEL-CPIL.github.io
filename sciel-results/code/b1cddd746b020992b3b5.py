import math
import time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """Returns vertices of a regular hexagon centered at (cx, cy) with side s and rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point is inside hexagon. Returns True if inside."""
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-a), math.sin(-a)
    rx = dx * ca - dy * sa
    ry = dx * sa + dy * ca
    apothem = s * math.sqrt(3) / 2
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if abs(proj) > apothem + 1e-9:
            return False
    return True

def _hexagon_distance(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg) for unit hexagons.
    Returns minimum distance between the two hexagons (0 if overlapping or touching).
    Uses support function / linear programming approach.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    # Quick reject
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.0:
        return math.sqrt(dx*dx + dy*dy) - 2.0  # approximate lower bound
    
    # Compute all vertices
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    # Check if any vertex of one is inside the other
    for pt in v1:
        if _point_in_hex(pt[0], pt[1], c2x, c2y, 1.0, a2):
            return 0.0
    for pt in v2:
        if _point_in_hex(pt[0], pt[1], c1x, c1y, 1.0, a1):
            return 0.0
    
    # Check edge intersections
    def seg_intersect(p1, p2, p3, p4):
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
    
    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return 0.0
    
    # If no overlap, compute minimum distance between edges
    min_dist = float('inf')
    for i in range(6):
        for j in range(6):
            # Distance between segment v1[i]-v1[i+1] and v2[j]-v2[j+1]
            # Use point-to-segment distance for endpoints
            pts1 = [v1[i], v1[(i+1)%6]]
            pts2 = [v2[j], v2[(j+1)%6]]
            for p in pts1:
                for q in pts2:
                    d = math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
                    if d < min_dist:
                        min_dist = d
            # Also check perpendicular distances (approximate)
            # For convex polygons, min distance is between a vertex of one and edge of other
    # More precise: vertex-to-edge distances
    def point_to_segment_dist(px, py, ax, ay, bx, by):
        dx, dy = bx-ax, by-ay
        if dx*dx + dy*dy == 0:
            return math.sqrt((px-ax)**2 + (py-ay)**2)
        t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        qx, qy = ax + t*dx, ay + t*dy
        return math.sqrt((px-qx)**2 + (py-qy)**2)
    
    for i in range(6):
        for j in range(6):
            d = point_to_segment_dist(v1[i][0], v1[i][1], v2[j][0], v2[j][1], v2[(j+1)%6][0], v2[(j+1)%6][1])
            if d < min_dist:
                min_dist = d
            d = point_to_segment_dist(v2[j][0], v2[j][1], v1[i][0], v1[i][1], v1[(i+1)%6][0], v1[(i+1)%6][1])
            if d < min_dist:
                min_dist = d
    
    return min_dist

def _greedy_place(seed, grid_angle_offset=0.0, spacing_factor=1.0):
    """
    Greedy placement of N unit hexagons using a lattice grid.
    Returns list of (cx, cy, angle_deg) or None if failed.
    """
    import random
    rng = random.Random(seed)
    
    # Generate candidate positions on a hexagonal lattice
    # Hex lattice with spacing 2*apothem = sqrt(3) for touching hexagons
    # But we need disjoint (non-overlapping), so spacing >= 2*apothem = sqrt(3)
    # Actually for regular hexagons of side 1, the distance between centers
    # for non-overlapping is at least 2 * apothem = sqrt(3) ~ 1.732
    # But with rotation, it can be less in some directions.
    # Safe minimum center distance: 2 * apothem = sqrt(3)
    
    min_center_dist = math.sqrt(3) * spacing_factor
    
    # Generate grid points
    # Use a hexagonal grid
    grid_spacing = min_center_dist * 1.01  # slight extra to avoid touching
    
    # Generate points in a bounding region
    max_radius = 8.0
    candidates = []
    
    # Hexagonal lattice generation
    # Basis vectors for hex grid
    for q in range(-int(max_radius/grid_spacing), int(max_radius/grid_spacing)+1):
        for r in range(-int(max_radius/grid_spacing), int(max_radius/grid_spacing)+1):
            # Axial coordinates to Cartesian
            x = grid_spacing * (q + r/2.0)
            y = grid_spacing * (math.sqrt(3)/2.0 * r)
            # Apply grid rotation
            a = math.radians(grid_angle_offset)
            xr = x * math.cos(a) - y * math.sin(a)
            yr = x * math.sin(a) + y * math.cos(a)
            if xr*xr + yr*yr <= max_radius*max_radius:
                candidates.append((xr, yr))
    
    # Shuffle candidates for diversity
    rng.shuffle(candidates)
    
    placed = []
    for (cx, cy) in candidates:
        # Check against all placed hexagons
        ok = True
        for (pcx, pcy, pang) in placed:
            dist = math.sqrt((cx-pcx)**2 + (cy-pcy)**2)
            if dist < min_center_dist - 1e-6:
                ok = False
                break
            # More precise check
            if dist < 2.0:
                d = _hexagon_distance((cx, cy, 0.0), (pcx, pcy, pang))
                if d < 1e-4:
                    ok = False
                    break
        if ok:
            placed.append((cx, cy, 0.0))
            if len(placed) == N:
                break
    
    if len(placed) < N:
        return None
    return placed

def _min_enclosing_hexagon(inner_hexes):
    """
    Given a list of (cx, cy, angle_deg) for unit hexagons,
    find the minimum side length regular hexagon (with free center and rotation)
    that contains all of them.
    
    Uses linear programming: for a regular hexagon with center (ox, oy), side s, 
    and rotation angle theta, a point (px, py) is inside iff for all 6 face normals n_i,
    n_i . (p - o) <= s * apothem_unit
    
    where apothem_unit = sqrt(3)/2 for unit side.
    
    So: n_i(theta) . (p - o) <= s * sqrt(3)/2  for all i, all p.
    
    Let u = s * sqrt(3)/2 (the apothem of the outer hexagon).
    We want to minimize u (equivalently s = 2u/sqrt(3)).
    
    Variables: ox, oy, u, and the 6 normal directions are determined by theta.
    This is nonlinear in theta. So we need to search over theta.
    
    For a fixed theta, the problem becomes:
    Minimize u
    Subject to: for all inner vertices v, and for all 6 normals n_i(theta):
        n_i(theta) . (v - (ox,oy)) <= u
        i.e., n_i(theta).v - n_i(theta).o <= u
    
    This is linear in (ox, oy, u) for fixed theta!
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang) in inner_hexes:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.extend(verts)
    
    if len(all_verts) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    best_side = float('inf')
    best_ox, best_oy, best_theta = 0.0, 0.0, 0.0
    
    # Search over theta (outer hexagon rotation)
    # By symmetry, we only need theta in [0, 60)
    # Use a fine grid + local refinement
    
    theta_vals = np.linspace(0, 60, 37)  # every 1.5 degrees
    
    for theta_deg in theta_vals:
        theta = math.radians(theta_deg)
        
        # 6 face normals of the outer hexagon
        # For a regular hexagon with vertex at angle theta, the face normals are at theta + 30 + k*60
        normals = []
        for k in range(3):
            ang_n = theta + math.radians(30 + k*60)
            nx = math.cos(ang_n)
            ny = math.sin(ang_n)
            normals.append((nx, ny))
        
        # For LP: variables are ox, oy, u
        # Constraints: for each vertex v and each normal n:
        #   n.x * ox + n.y * oy + u >= n.x * v.x + n.y * v.y
        #   (since n . (v - o) <= u  =>  n.v - n.o <= u  =>  n.v <= n.o + u)
        #   => -n.x*ox - n.y*oy + u >= -(n.x*v.x + n.y*v.y) ... let me re-derive
        #
        # n . (v - o) <= u
        # n.v - n.o <= u
        # -n.o + (-1)*u <= -n.v  ... no
        # n.v <= n.o + u
        # n.v - n.o - u <= 0
        # In standard form for linprog (minimize c^T x s.t. A x <= b):
        # Let x = [ox, oy, u]
        # Constraint: n.x*ox + n.y*oy + u >= n.x*v.x + n.y*v.y
        # => -n.x*ox - n.y*oy - u <= -(n.x*v.x + n.y*v.y)
        
        n_constraints = len(all_verts) * 3  # 3 pairs of normals (6 faces, but we use 3 axes)
        # Actually we need 6 constraints per vertex (one per face), but 3 pairs is equivalent
        # because the hexagon is the intersection of 3 strips.
        # Wait: a regular hexagon is the intersection of 3 strips (6 half-planes).
        # The 3 normals I computed are for 3 of the 6 faces. The other 3 are their negatives.
        # So we need both n and -n for each of the 3 directions.
        
        A_rows = []
        b_vals = []
        
        for v in all_verts:
            vx, vy = v
            for (nx, ny) in normals:
                # n . (v - o) <= u  =>  -n.x*ox - n.y*oy - u <= -n.v
                A_rows.append([-nx, -ny, -1.0])
                b_vals.append(-(nx*vx + ny*vy))
                # -n . (v - o) <= u  =>  n.x*ox + n.y*oy - u <= n.v
                A_rows.append([nx, ny, -1.0])
                b_vals.append(nx*vx + ny*vy)
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        c = np.array([0.0, 0.0, 1.0])  # minimize u
        
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.success:
                u = res.x[2]
                side = 2.0 * u / math.sqrt(3)
                ox, oy = res.x[0], res.x[1]
                if side < best_side:
                    best_side = side
                    best_ox, best_oy, best_theta = ox, oy, theta_deg
        except Exception:
            pass
    
    # Refine around best theta
    if best_side < float('inf'):
        for dt in np.linspace(-2.0, 2.0, 41):
            theta_deg = best_theta + dt
            if theta_deg < 0 or theta_deg >= 60:
                continue
            theta = math.radians(theta_deg)
            normals = []
            for k in range(3):
                ang_n = theta + math.radians(30 + k*60)
                nx = math.cos(ang_n)
                ny = math.sin(ang_n)
                normals.append((nx, ny))
            
            A_rows = []
            b_vals = []
            for v in all_verts:
                vx, vy = v
                for (nx, ny) in normals:
                    A_rows.append([-nx, -ny, -1.0])
                    b_vals.append(-(nx*vx + ny*vy))
                    A_rows.append([nx, ny, -1.0])
                    b_vals.append(nx*vx + ny*vy)
            
            A = np.array(A_rows)
            b = np.array(b_vals)
            c = np.array([0.0, 0.0, 1.0])
            
            try:
                res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
                if res.success:
                    u = res.x[2]
                    side = 2.0 * u / math.sqrt(3)
                    if side < best_side:
                        best_side = side
                        best_ox, best_oy, best_theta = res.x[0], res.x[1], theta_deg
            except Exception:
                pass
    
    return best_ox, best_oy, best_side, best_theta

def _validate(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are disjoint and inside the outer hexagon."""
    ocx, ocy = outer_center
    
    # Check containment
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for (vx, vy) in verts:
            if not _point_in_hex(vx, vy, ocx, ocy, outer_side, outer_angle):
                return False
    
    # Check pairwise disjointness
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            d = _hexagon_distance(inner[i], inner[j])
            if d < -1e-4:  # overlapping
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_result = None
    best_side = float('inf')
    
    # Phase 1: Multiple greedy placements with different parameters
    seeds = list(range(50))
    grid_angles = [0, 10, 20, 30, 40, 50, 15, 25, 35, 45, 55, 5, 12, 22, 32, 42, 52, 7, 17, 27, 37, 47, 57, 2, 8, 18, 28, 38, 48, 58]
    spacing_factors = [1.0, 1.01, 1.02, 1.03, 1.05, 1.08, 1.1, 1.15, 1.2]
    
    attempt = 0
    for seed in seeds:
        if time.time() > deadline - 20.0:
            break
        
        for gangle in grid_angles:
            if time.time() > deadline - 20.0:
                break
            for sf in spacing_factors:
                if time.time() > deadline - 20.0:
                    break
                
                placed = _greedy_place(seed * 1000 + gangle * 10 + int(sf*100), gangle, sf)
                if placed is None:
                    continue
                
                # Compute minimum enclosing hexagon
                ox, oy, side, oang = _min_enclosing_hexagon(placed)
                
                if side < best_side:
                    # Validate
                    if _validate(placed, (ox, oy), side, oang):
                        best_side = side
                        best_result = (placed, (ox, oy), side, oang)
        
        # Also try a few more creative placements
        if time.time() > deadline - 20.0:
            break
    
    # Phase 2: Local refinement of best result
    if best_result is not None and time.time() < deadline - 5.0:
        inner, (ox, oy), side, oang = best_result
        # Try small perturbations of inner positions
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            perturbed = []
            for (cx, cy, ang) in inner:
                dx = (np.random.rand() - 0.5) * 0.1
                dy = (np.random.rand() - 0.5) * 0.1
                perturbed.append((cx + dx, cy + dy, ang))
            
            # Check validity of perturbed config
            valid = True
            for i in range(len(perturbed)):
                for j in range(i+1, len(perturbed)):
                    d = _hexagon_distance(perturbed[i], perturbed[j])
                    if d < 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_ox, new_oy, new_side, new_oang = _min_enclosing_hexagon(perturbed)
                if new_side < best_side:
                    if _validate(perturbed, (new_ox, new_oy), new_side, new_oang):
                        best_side = new_side
                        best_result = (perturbed, (new_ox, new_oy), new_side, new_oang)
    
    # Fallback: if nothing found, use a simple valid configuration
    if best_result is None:
        # Place hexagons in a line with spacing sqrt(3)
        inner = []
        for i in range(N):
            inner.append((i * math.sqrt(3), 0.0, 0.0))
        ox, oy, side, oang = _min_enclosing_hexagon(inner)
        best_result = (inner, (ox, oy), side, oang)
        best_side = side
    
    inner, (ox, oy), side, oang = best_result
    
    # Format output
    inner_list = [[cx, cy, ang] for (cx, cy, ang) in inner]
    return inner_list, (ox, oy), side, oang

if __name__ == "__main__":
    r = construct()
    print(r)
