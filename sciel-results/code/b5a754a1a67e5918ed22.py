"""Rewritten program: physics-based repulsion + binary search on container size."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
_EPS = 1e-9

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotated by angle_deg."""
    a = np.radians(angle_deg)
    # Unit hexagon vertices (flat-top orientation at 0 deg): angles 0,60,120,180,240,300
    angles = a + np.radians(np.arange(6) * 60.0)
    verts = side * np.column_stack([np.cos(angles), np.sin(angles)])
    return verts + np.array([cx, cy])

def _hex_face_normals(angle_deg):
    """Return the 6 outward unit normals of a hexagon rotated by angle_deg.
    For a flat-top hexagon (vertices at 0,60,...,300 deg), the face normals point at 30,90,150,210,270,330 deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(30.0 + np.arange(6) * 60.0)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides, outer_angle=0.0):
    """Find minimum side length of a regular hexagon (centered at origin, rotated by outer_angle)
    that contains all inner hexagons.
    
    inner_positions: list of (x,y)
    inner_angles: list of angle_degrees
    inner_sides: list of side lengths
    outer_angle: rotation of outer hexagon in degrees
    
    Returns: (center_x, center_y, side_length)
    """
    n = len(inner_positions)
    
    # For each inner hexagon, compute the support function in each of the 6 outer hexagon directions
    # The outer hexagon with center (cx,cy), side s, rotation theta contains point p iff:
    #   for all 6 face directions d_i: d_i . (p - center) <= s * apothem_factor
    # where apothem = s * sqrt(3)/2, so the constraint is d_i . (p - center) <= s * sqrt(3)/2
    # i.e., s >= max_i [ d_i . (p - center) ] / (sqrt(3)/2) for all p in all inner hexagons
    
    outer_normals = _hex_face_normals(outer_angle)  # 6x2
    sqrt3 = np.sqrt(3.0)
    
    # For each inner hexagon, the max of d_i . p over p in the hexagon is:
    #   d_i . center_inner + max over vertices... actually use support function
    # For a regular hexagon with center c, side s, rotation a:
    # support in direction d = d . c + s * max_k (d . v_k) where v_k are unit-hex vertices
    # The max of d . v over a unit hexagon (vertices at angles 0,60,...,300 rotated by a)
    # equals the max over the 6 vertex directions.
    
    # Precompute: for each inner hex, for each outer normal direction, compute max projection
    # max_{p in hex_j} d_i . p = d_i . c_j + s_j * max_k (d_i . (R(a_j) * u_k))
    # where u_k are unit hex vertices (side 1)
    
    # Unit hex vertices for side 1
    unit_angles = np.radians(np.arange(6) * 60.0)
    unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])  # 6x2
    
    # For each inner hex j and each outer normal i:
    # max_proj[j][i] = d_i . c_j + s_j * max_k (d_i . R(a_j) u_k)
    # R(a) u_k = rotate u_k by a
    
    # Compute all rotations of unit vertices for each inner hex
    max_projs = np.zeros((n, 6))
    for j in range(n):
        cx_j, cy_j = inner_positions[j]
        a_j = np.radians(inner_angles[j])
        s_j = inner_sides[j]
        # Rotate unit vertices by a_j
        cos_a, sin_a = np.cos(a_j), np.sin(a_j)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_verts = unit_verts @ rot_mat.T  # 6x2
        # For each outer normal d_i, compute max_k d_i . rot_verts[k]
        for i in range(6):
            d_i = outer_normals[i]
            projs = rot_verts @ d_i  # 6 values
            max_p = np.max(projs)
            max_projs[j, i] = d_i[0] * cx_j + d_i[1] * cy_j + s_j * max_p
    
    # Now we need to find center (cx, cy) and side s such that:
    #   for all j, i: d_i . (p - center) <= s * sqrt(3)/2  for all p in hex_j
    #   i.e., max_{p in hex_j} d_i . p - d_i . center <= s * sqrt(3)/2
    #   i.e., max_projs[j,i] - d_i[0]*cx - d_i[1]*cy <= s * sqrt(3)/2
    #   i.e., s >= (max_projs[j,i] - d_i[0]*cx - d_i[1]*cy) / (sqrt(3)/2)
    #   for all j, i
    # Also the "back" side: we need the hexagon to contain, so we also need:
    #   -d_i . (p - center) <= s * sqrt(3)/2  (the opposite face)
    #   i.e., d_i . center - max_{p in hex_j} d_i . p <= s * sqrt(3)/2
    #   i.e., s >= (d_i[0]*cx + d_i[1]*cy - max_projs[j,i]) / (sqrt(3)/2)
    # Wait, that's not right. Let me reconsider.
    
    # A regular hexagon centered at C with side s and rotation theta has the property that
    # for each of the 6 face normals d_i (unit vectors), the face is at distance s*sqrt(3)/2 from center.
    # So a point p is inside iff for all i: d_i . (p - C) <= s * sqrt(3)/2
    # AND also we need the opposite: actually for a hexagon, the 6 face normals already cover all 6 sides.
    # The hexagon is the intersection of 6 half-planes: d_i . (p - C) <= s*sqrt(3)/2 for i=0..5.
    # But wait, that's only 6 half-planes and they define a hexagon. Let me verify:
    # For a regular hexagon with flat top (normal at 90 deg = (0,1)), the top face is at y = s*sqrt(3)/2.
    # The bottom face has normal (0,-1) at 270 deg, so the constraint is (0,-1). (p-C) <= s*sqrt(3)/2, i.e., C_y - p_y <= s*sqrt(3)/2, i.e., p_y >= C_y - s*sqrt(3)/2.
    # Yes, that's correct. So 6 constraints with the 6 normals suffice.
    
    # So: s >= max_{j,i} [ (max_projs[j,i] - d_i . C) / (sqrt(3)/2) ]
    #     s >= max_{j,i} [ (d_i . C - min_projs[j,i]) / (sqrt(3)/2) ]
    # where min_projs[j,i] = min_{p in hex_j} d_i . p = d_i . c_j + s_j * min_k (d_i . R(a_j) u_k)
    
    # Actually, we can combine: the constraint is |d_i . (p - C)| <= s*sqrt(3)/2 is NOT what we want.
    # We want d_i . (p - C) <= s*sqrt(3)/2 for all i and all p in all hexagons.
    # This gives: s >= (max_{j,i} (max_projs[j,i] - d_i . C)) / (sqrt(3)/2)
    # But we also need the hexagon to be bounded, which it is by the 6 constraints.
    # However, we also need: for the hexagon to contain the point, we need ALL 6 half-plane constraints.
    # The 6 constraints with normals d_0..d_5 (at 30,90,...,330 deg) define the hexagon.
    # So the condition is: for all j, all i: max_{p in hex_j} d_i . p - d_i . C <= s * sqrt(3)/2
    # i.e., s >= (max_{j} max_projs[j,i] - d_i . C) / (sqrt(3)/2) for each i
    # And s must satisfy all 6 of these.
    
    # So: s(C) = max_i [ (max_j max_projs[j,i] - d_i . C) / (sqrt(3)/2) ]
    # And we want to minimize s(C) over C.
    
    # This is a convex optimization problem! s(C) is the max of 6 affine functions of C, so it's convex.
    # We can use scipy minimize or solve it directly.
    
    # Let's define f(C) = max_i [ (M_i - d_i . C) / (sqrt(3)/2) ] where M_i = max_j max_projs[j,i]
    # We want to minimize f(C).
    
    M = np.max(max_projs, axis=0)  # 6 values: max over j for each i
    
    def objective(C):
        cx, cy = C
        vals = np.zeros(6)
        for i in range(6):
            vals[i] = (M[i] - outer_normals[i][0]*cx - outer_normals[i][1]*cy) / (sqrt3/2.0)
        return np.max(vals)
    
    # Use Nelder-Mead to minimize
    x0 = np.array([0.0, 0.0])
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
    C_opt = res.x
    s_opt = objective(C_opt)
    
    return (C_opt[0], C_opt[1], s_opt)

def _check_validity(inner_positions, inner_angles, inner_sides, outer_center, outer_side, outer_angle):
    """Check that all inner hexagons are inside the outer hexagon and disjoint."""
    n = len(inner_positions)
    
    # Check containment
    outer_normals = _hex_face_normals(outer_angle)
    sqrt3 = np.sqrt(3.0)
    apothem = outer_side * sqrt3 / 2.0
    
    unit_angles = np.radians(np.arange(6) * 60.0)
    unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])
    
    for j in range(n):
        cx_j, cy_j = inner_positions[j]
        a_j = np.radians(inner_angles[j])
        s_j = inner_sides[j]
        cos_a, sin_a = np.cos(a_j), np.sin(a_j)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_verts = unit_verts @ rot_mat.T
        # Vertices of inner hexagon j
        verts_j = s_j * rot_verts + np.array([cx_j, cy_j])
        for v in verts_j:
            for i in range(6):
                d_i = outer_normals[i]
                val = d_i[0]*(v[0]-outer_center[0]) + d_i[1]*(v[1]-outer_center[1])
                if val > apothem + 1e-6:
                    return False
    
    # Check disjointness: for each pair, check if hexagons overlap
    # Two convex polygons overlap if one vertex is inside the other or edges intersect
    # Simpler: check if any vertex of one is inside the other
    def point_in_hex(px, py, cx_h, cy_h, angle_deg_h, side_h):
        a_h = np.radians(angle_deg_h)
        normals_h = _hex_face_normals(angle_deg_h)
        apothem_h = side_h * sqrt3 / 2.0
        for i in range(6):
            d_i = normals_h[i]
            val = d_i[0]*(px-cx_h) + d_i[1]*(py-cy_h)
            if val > apothem_h + 1e-9:
                return False
        return True
    
    for j in range(n):
        for k in range(j+1, n):
            cx_j, cy_j = inner_positions[j]
            cx_k, cy_k = inner_positions[k]
            # Get vertices of hex j
            a_j = np.radians(inner_angles[j])
            s_j = inner_sides[j]
            cos_a, sin_a = np.cos(a_j), np.sin(a_j)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rot_verts = unit_verts @ rot_mat.T
            verts_j = s_j * rot_verts + np.array([cx_j, cy_j])
            
            a_k = np.radians(inner_angles[k])
            s_k = inner_sides[k]
            cos_a, sin_a = np.cos(a_k), np.sin(a_k)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rot_verts = unit_verts @ rot_mat.T
            verts_k = s_k * rot_verts + np.array([cx_k, cy_k])
            
            # Check if any vertex of j is inside k
            for v in verts_j:
                if point_in_hex(v[0], v[1], cx_k, cy_k, inner_angles[k], s_k):
                    return False
            # Check if any vertex of k is inside j
            for v in verts_k:
                if point_in_hex(v[0], v[1], cx_j, cy_j, inner_angles[j], s_j):
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = 150.0
    
    inner_sides = [1.0] * N
    
    # Initial guess: honeycomb-like arrangement
    # Place hexagons in a rough grid
    hex_spacing = 2.0  # distance between centers for touching hexagons (side 1)
    
    # Use a hexagonal lattice pattern
    positions = []
    # 4 rows of 3, or 3 rows of 4
    # Let's try 3x4 grid
    for row in range(3):
        for col in range(4):
            x = col * hex_spacing * 1.5
            y = row * hex_spacing * np.sqrt(3)
            if row % 2 == 1:
                x += hex_spacing * 0.75
            positions.append([x, y, 0.0])
    
    # Center them
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (max(xs) + min(xs)) / 2.0
    cy_mean = (max(ys) + min(ys)) / 2.0
    for p in positions:
        p[0] -= cx_mean
        p[1] -= cy_mean
    
    # Binary search on outer side length
    # Lower bound: at least sqrt(3) (one hexagon)
    # Upper bound: large
    lo = 3.0
    hi = 20.0
    
    best_result = None
    best_side = hi
    
    # We'll do a few iterations of binary search
    for iteration in range(8):
        if time.time() > deadline - 10.0:
            break
        
        mid = (lo + hi) / 2.0
        
        # Try to find a valid configuration with outer side = mid
        # Use optimization to minimize overlap while keeping containment
        def penalty_function(params):
            """params: 3*N values: x1,y1,ang1, x2,y2,ang2, ..."""
            elapsed = time.time() - start_time
            if elapsed > deadline - 5.0:
                return 1e10
            
            total_penalty = 0.0
            pos = []
            angs = []
            for j in range(N):
                x = params[3*j]
                y = params[3*j+1]
                ang = params[3*j+2]
                pos.append((x, y))
                angs.append(ang)
            
            # Containment penalty
            outer_center = (0.0, 0.0)
            outer_angle = 0.0
            outer_normals = _hex_face_normals(outer_angle)
            sqrt3 = np.sqrt(3.0)
            apothem = mid * sqrt3 / 2.0
            
            unit_angles = np.radians(np.arange(6) * 60.0)
            unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])
            
            for j in range(N):
                x_j, y_j = pos[j]
                a_j = np.radians(angs[j])
                s_j = 1.0
                cos_a, sin_a = np.cos(a_j), np.sin(a_j)
                rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rot_verts = unit_verts @ rot_mat.T
                verts_j = s_j * rot_verts + np.array([x_j, y_j])
                for v in verts_j:
                    for i in range(6):
                        d_i = outer_normals[i]
                        val = d_i[0]*(v[0]-outer_center[0]) + d_i[1]*(v[1]-outer_center[1])
                        if val > apothem:
                            total_penalty += (val - apothem) ** 2
            
            # Overlap penalty
            for j in range(N):
                for k in range(j+1, N):
                    x_j, y_j = pos[j]
                    x_k, y_k = pos[k]
                    dist = np.sqrt((x_j-x_k)**2 + (y_j-y_k)**2)
                    # Minimum distance for non-overlap depends on angles, but use a lower bound
                    # Two unit hexagons need center distance >= sqrt(3) if aligned, but can be less if rotated
                    # Use a conservative estimate: if dist < 1.5, they likely overlap
                    if dist < 1.7:
                        overlap = max(0, 1.7 - dist)
                        total_penalty += overlap ** 2 * 10.0
            
            return total_penalty
        
        # Start from current positions
        x0 = np.zeros(3*N)
        for j in range(N):
            x0[3*j] = positions[j][0]
            x0[3*j+1] = positions[j][1]
            x0[3*j+2] = positions[j][2]
        
        res = minimize(penalty_function, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if time.time() > deadline - 5.0:
            break
        
        # Check if valid
        new_positions = []
        new_angles = []
        for j in range(N):
            new_positions.append((res.x[3*j], res.x[3*j+1]))
            new_angles.append(res.x[3*j+2])
        
        if _check_validity(new_positions, new_angles, inner_sides, (0.0, 0.0), mid, 0.0):
            hi = mid
            best_side = mid
            best_result = (new_positions, new_angles)
            # Update positions for next iteration
            for j in range(N):
                positions[j][0] = new_positions[j][0]
                positions[j][1] = new_positions[j][1]
                positions[j][2] = new_angles[j]
        else:
            lo = mid
    
    # Return the best result found
    if best_result is not None:
        inner = [[best_result[0][j][0], best_result[0][j][1], best_result[1][j]] for j in range(N)]
        return inner, (0.0, 0.0), best_side, 0.0
    else:
        # Fallback: return initial guess with a large container
        inner = [[p[0], p[1], p[2]] for p in positions]
        return inner, (0.0, 0.0), hi, 0.0
