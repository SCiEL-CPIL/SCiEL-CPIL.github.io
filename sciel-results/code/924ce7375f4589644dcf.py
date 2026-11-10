# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60) * np.arange(6)
    xs = cx + d * np.cos(angles)
    ys = cy + d * np.sin(angles)
    return np.column_stack([xs, ys])


def _edge_normals(vertices):
    """Return outward normals for each edge of a polygon given in CCW order."""
    n = len(vertices)
    normals = np.zeros_like(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Outward normal for CCW polygon: rotate edge by -90 degrees (or +90 depending on orientation)
        # For CCW polygon, outward normal is (edge_y, -edge_x) normalized? 
        # Let's check: edge (1,0) -> normal (0,-1) points down. For a unit square CCW:
        # (0,0)->(1,0)->(1,1)->(0,1). Edge (1,0) at bottom. Outward is (0,-1). So (edge_y, -edge_x) = (0, -1). Correct.
        # Wait, for a hexagon centered at origin with vertex at (1,0):
        # Vertices: (1,0), (0.5, sqrt(3)/2), ... CCW.
        # Edge from (1,0) to (0.5, sqrt(3)/2) is (-0.5, sqrt(3)/2).
        # Outward normal should point away from origin.
        # Normal (-0.5, sqrt(3)/2) rotated -90: (sqrt(3)/2, 0.5). This points into the first quadrant, away from origin. Correct.
        # So normal = (edge_y, -edge_x) is NOT correct for outward? 
        # Let's re-evaluate. For CCW polygon, interior is to the left of each edge.
        # So outward normal is to the right of the edge direction.
        # Rotating a vector (x, y) by -90 degrees gives (y, -x).
        # So normal = (edge[1], -edge[0]) is the right-hand normal, which is outward for CCW.
        norm = np.array([edge[1], -edge[0]])
        norm /= np.linalg.norm(norm)
        normals[i] = norm
    return normals


def _min_enclosing_hex_side(inner_hexes):
    """
    Given a list of (cx, cy, angle_deg) for unit hexagons, 
    find the minimum side length 's' of a regular hexagon (centered at origin, angle 0) 
    that contains all of them.
    Returns s, center_x, center_y, angle_deg of the outer hexagon.
    """
    def objective(params):
        cx, cy, angle_deg = params
        a = np.deg2rad(angle_deg)
        
        # Compute the maximum projection of all inner hexagon vertices 
        # onto the 6 outward normals of the outer hexagon.
        # For the outer hexagon to contain a point v, for each normal n_i:
        # dot(v - center, n_i) <= s * cos(30 deg) = s * sqrt(3)/2
        # So s >= max(dot(v - center, n_i)) / (sqrt(3)/2)
        
        max_proj = 0
        for (ix, iy, iangle) in inner_hexes:
            # Vertices of this inner hexagon
            ia = np.deg2rad(iangle)
            i_angles = ia + np.deg2rad(60) * np.arange(6)
            vs = np.column_stack([
                ix + np.cos(i_angles),
                iy + np.sin(i_angles)
            ])
            
            # Shift relative to outer center
            vs_rel = vs - np.array([cx, cy])
            
            # Outer normals
            o_angles = a + np.deg2rad(60) * np.arange(6)
            # Outward normals for CCW hexagon:
            # Vertex i is at angle o_angles[i]. Edge i is from v_i to v_{i+1}.
            # Normal for edge i points outward.
            # The direction of the normal for edge i is o_angles[i] + 30 deg? 
            # Let's derive. Center at origin. Vertex i: (cos(theta_i), sin(theta_i)).
            # Edge i connects theta_i to theta_{i+1}. Midpoint of edge is at angle (theta_i + theta_{i+1})/2 = theta_i + 30 deg.
            # The outward normal is in the direction of this midpoint.
            # So normal_i = (cos(theta_i + 30), sin(theta_i + 30)).
            n_angles = a + np.deg2rad(60) * np.arange(6) + np.deg2rad(30)
            normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
            
            # Projections
            projs = vs_rel @ normals.T  # shape (6, 6) -> actually (6 inner verts, 6 normals)
            # We need the max over all inner vertices and all 6 normals.
            # But actually, for a convex polygon to contain another, we just need to check the support function.
            # The max projection of the inner hexagon onto normal n is:
            # max over vertices v of dot(v, n).
            # So for each normal n_j, find max dot(v, n_j) over all 12*6 vertices.
            # This is equivalent to max over all vertices of dot(v, n_j).
            # We can compute dot(v, n_j) for all v and all j, then take max over v for each j, then max over j.
            # Or just take the global max of all dot products? 
            # No, s is determined by the max over j of (max_v dot(v, n_j)) / (sqrt(3)/2).
            # So we need for each j: max_v dot(v, n_j). Then s = max_j (that value) / (sqrt(3)/2).
            
            col_max = np.max(projs, axis=0)  # max over 6 vertices for each of 6 normals
            global_max = np.max(col_max)
            if global_max > max_proj:
                max_proj = global_max
        
        return max_proj / (np.sqrt(3) / 2)
    
    # Use Nelder-Mead to minimize s
    # Initial guess: center at mean of inner hex centers, angle 0
    centers = np.array([(x, y) for (x, y, a) in inner_hexes])
    cx0, cy0 = centers.mean(axis=0)
    x0 = np.array([cx0, cy0, 0.0])
    
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200})
    
    cx, cy, angle_deg = res.x
    s = objective(res.x)
    
    return s, cx, cy, angle_deg


def _check_separation(hex1, hex2, tol=1e-6):
    """
    Check if two hexagons are separated (non-overlapping) using Separating Axis Theorem.
    hex1, hex2: (cx, cy, angle_deg)
    Returns True if separated.
    """
    v1 = _hex_vertices(*hex1)
    v2 = _hex_vertices(*hex2)
    
    normals1 = _edge_normals(v1)
    normals2 = _edge_normals(v2)
    
    all_normals = np.vstack([normals1, normals2])
    
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 - tol or max2 < min1 - tol:
            return True  # Separated
        
        # If they overlap on this axis, continue to next axis
        # If they touch (gap <= 0), it's not strictly separated, but SAT says if NO axis has a gap, they overlap/touch.
        # The task says "two inner hexagons not separated: some edge normal ... must give projection intervals more than 1e-6 apart".
        # So if for ALL normals, the gap is <= 1e-6, they are "not separated" (invalid).
        # We want them to be separated, so there MUST exist a normal with gap > 1e-6.
        # Wait, the logic is: if they are NOT separated, then for EVERY normal, the gap is <= 1e-6.
        # So if we find a normal with gap > 1e-6, they ARE separated.
        # If we check ALL normals and none have gap > 1e-6, they are NOT separated.
        pass
    
    return False


def _vertex_inside_outer(v, outer_vertices, tol=1e-6):
    """
    Check if point v is inside or on the boundary of the outer hexagon (CCW).
    For every outer edge (p1, p2), cross(p2 - p1, v - p1) must be >= -tol.
    """
    for i in range(6):
        p1 = outer_vertices[i]
        p2 = outer_vertices[(i + 1) % 6]
        edge = p2 - p1
        rel = v - p1
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < -tol:
            return False
    return True


def _validate(inner_hexes, outer_hex, s):
    """
    Validate the packing.
    Returns True if valid.
    """
    # 1. s >= sqrt(12)
    if s < np.sqrt(12):
        return False
    
    # 2. Outer vertices distinct
    outer_verts = _hex_vertices(*outer_hex, s)
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # 3. Unit edge check for inner hexagons (implicit if we use side 1, but let's check)
    for hex_data in inner_hexes:
        verts = _hex_vertices(*hex_data)
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i+1)%6]
            if abs(np.linalg.norm(p2 - p1) - 1) > 1e-8:
                return False
    
    # 4. Separation between inner hexagons
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_separation(inner_hexes[i], inner_hexes[j]):
                return False
    
    # 5. Inner vertices inside outer hexagon
    # "after shifting all centers by minus the outer center"
    outer_center = np.array([outer_hex[0], outer_hex[1]])
    for hex_data in inner_hexes:
        verts = _hex_vertices(*hex_data)
        # Shift vertices by -outer_center? 
        # The condition says: "an inner vertex v outside the outer hexagon: for every outer edge (p1, p2) in vertex order, cross(p2 - p1, v - p1) must be >= -1e-6"
        # This implies the outer hexagon is defined by its absolute coordinates (outer_hex[0], outer_hex[1], angle) and side s.
        # And v is the absolute coordinate of the inner vertex.
        # So we just check v against the absolute outer hexagon.
        for v in verts:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
    return True


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep safety margin
    
    # Initial configuration: Honeycomb-like arrangement
    # Center (0,0)
    # 6 around it at distance sqrt(3) (which is 2 * apothem of unit hex)
    # Wait, distance between centers of touching unit hexagons is sqrt(3).
    # Apothem of unit hex is sqrt(3)/2. Distance between centers = 2 * apothem = sqrt(3).
    
    inner_hexes = [
        (0, 0, 0),  # center
    ]
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        inner_hexes.append((x, y, 0))
    
    # Add 5 more to reach 12?
    # Next ring is at distance 2*sqrt(3)?
    # Let's try a compact arrangement.
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # Spacing: x_dist = 1.5, y_dist = sqrt(3)/2 * 2 = sqrt(3)?
    # No, for grid:
    # (0,0), (1.5, sqrt(3)/2), (3, 0) -> Row 0
    # (0.75, -sqrt(3)/4), (2.25, -sqrt(3)/4) -> No, standard grid is:
    # x = i * 1.5 + (j % 2) * 0.75
    # y = j * sqrt(3)/2
    
    # Let's stick to the first 7 (center + 6 neighbors) and add 5 more in the second ring.
    # The second ring centers are at distance 2*sqrt(3) from origin?
    # Actually, the lattice points are at (i * 1.5 + j * 0.75, j * sqrt(3)/2).
    # Distance squared = (1.5i + 0.75j)^2 + (sqrt(3)/2 j)^2
    # Let's just pick 5 more from the next layer.
    # Layer 2 (distance 2*sqrt(3) from center? No, distance to center is not uniform in Cartesian).
    
    # Let's use a simple grid of 12 hexagons.
    # 4 rows of 3.
    inner_hexes = []
    for r in range(4):
        for c in range(3):
            x = c * 1.5 + (r % 2) * 0.75
            y = r * np.sqrt(3) / 2
            inner_hexes.append((x, y, 0))
    
    # Center the grid
    centers = np.array([(x, y) for x, y, a in inner_hexes])
    mean_center = centers.mean(axis=0)
    inner_hexes = [(x - mean_center[0], y - mean_center[1], a) for x, y, a in inner_hexes]
    
    # Validate initial config
    if not _validate(inner_hexes, (0, 0, 0), 100):
        # Fallback to a very loose config
        inner_hexes = [(i*10, 0, 0) for i in range(12)]
    
    best_score = 1.0
    best_result = None
    best_inner = None
    best_outer = None
    best_s = None
    
    # Optimization
    # We can optimize the positions of the 12 hexagons to minimize the enclosing hexagon size.
    # Variables: 12 * 3 = 36 variables (x, y, angle for each).
    # Objective: Minimize s (side length of enclosing hexagon).
    
    def objective(params):
        inner_hexes_opt = []
        for i in range(12):
            x, y, a = params[3*i:3*i+3]
            inner_hexes_opt.append((x, y, a))
        
        s, cx, cy, angle_deg = _min_enclosing_hex_side(inner_hexes_opt)
        return s
    
    x0 = np.array([v for hex_data in inner_hexes for v in hex_data])
    
    # Run Nelder-Mead
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 1000})
    
    x_opt = res.x
    inner_hexes_opt = []
    for i in range(12):
        x, y, a = x_opt[3*i:3*i+3]
        inner_hexes_opt.append((x, y, a))
    
    s, cx, cy, angle_deg = _min_enclosing_hex_side(inner_hexes_opt)
    
    outer_hex = (cx, cy, angle_deg)
    
    # Validate
    if _validate(inner_hexes_opt, outer_hex, s):
        score = 1.0 / s
        if best_result is None or score > best_score:
            best_score = score
            best_result = (inner_hexes_opt, outer_hex, s)
    
    # If we have time, try to improve further with more iterations or different starts
    # But for now, return the best found.
    
    if best_result is not None:
        inner_hexes_final, outer_hex_final, s_final = best_result
    else:
        # Fallback to initial if optimization failed
        inner_hexes_final = inner_hexes
        outer_hex_final = (0, 0, 0)
        s_final = 100
        
        # Try to find a valid enclosing hex for the fallback
        s, cx, cy, angle_deg = _min_enclosing_hex_side(inner_hexes_final)
        if _validate(inner_hexes_final, (cx, cy, angle_deg), s):
            outer_hex_final = (cx, cy, angle_deg)
            s_final = s
    
    inner_array = np.array(inner_hexes_final)
    outer_array = np.array(outer_hex_final)
    
    return inner_array, outer_array, s_final


# EVOLVE-BLOCK-END
