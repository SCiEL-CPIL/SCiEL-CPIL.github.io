"""
Combined program for packing 12 unit hexagons in a minimum outer hexagon.
Uses SAT-based smooth penalties and multi-stage optimization (L-BFGS-B + Nelder-Mead).
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE_MARGIN = 10.0  # Seconds to leave before hard stop

def _hex_vertices(cx, cy, rot_deg, side=1.0):
    """
    Compute vertices of a regular hexagon.
    rot_deg: rotation in degrees.
    Returns (6, 2) array.
    """
    # Base angles for flat-top hexagon? Or pointy-top?
    # Standard math: angle 0 is (1,0).
    # Let's use standard orientation where one vertex is at angle 0 if rot=0?
    # Usually hexagons are defined with flat top or pointy top.
    # Let's define rot=0 as having a vertex at (side, 0).
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    """
    Compute outward normals to the edges of the hexagon.
    For a hexagon with vertices at angles 0, 60, ..., the edges are between these.
    Edge 0 is between 0 and 60 deg. Midpoint angle is 30 deg. Normal is 30 deg.
    """
    # Normals point to the midpoint of each edge.
    # If vertices are at 0, 60, 120..., edges are centered at 30, 90, 150...
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap_depth(v1, v2, n1, n2):
    """
    Calculate the minimum separation distance between two convex polygons.
    Positive if separated, negative if overlapping (penetration depth).
    Uses SAT (Separating Axis Theorem).
    v1, v2: (6,2) arrays of vertices.
    n1, n2: (6,2) arrays of edge normals.
    """
    min_overlap = float('inf')
    
    # Check axes from polygon 1
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            # Separating axis found
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    # Check axes from polygon 2
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            # Separating axis found
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
            
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    # No separating axis, polygons overlap
    return -min_overlap

def _outer_hex_side(points, outer_rot_deg):
    """
    Calculate the minimum side length of a regular hexagon with given rotation
    that contains all points.
    points: (M, 2) array of points.
    outer_rot_deg: rotation of the outer hexagon.
    """
    # The condition for a point (x,y) to be inside a regular hexagon centered at origin
    # with rotation theta and side length s is:
    # For each of the 6 face normals n_k, dot((x,y), n_k) <= s * sqrt(3) / 2
    # The face normals for a hexagon with vertex angle 0 are at 30, 90, 150...
    # Wait, if vertices are at 0, 60..., the faces are perpendicular to the bisectors.
    # Bisectors of edges (0-60) is 30. So normal is at 30.
    
    base_angles = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(outer_rot_deg)
    normals = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # Project all points onto each normal
    # points: (M, 2), normals: (6, 2)
    # proj: (M, 6)
    proj = points @ normals.T
    
    # We need max(abs(proj)) over all points and all normals
    # Since the hexagon is symmetric, we check the maximum projection in the direction of each normal.
    # Actually, we need the maximum dot product with each normal.
    # The constraint is max_k (max_p (p . n_k)) <= inradius
    # inradius = s * sqrt(3) / 2
    # s = 2 * inradius / sqrt(3)
    
    max_proj = np.max(np.abs(proj))
    inradius = max_proj
    side = 2.0 * inradius / np.sqrt(3.0)
    
    return side

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-7):
    """
    Check if the configuration is valid:
    1. No overlaps between inner hexagons.
    2. All inner hexagons are inside the outer hexagon.
    """
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
        
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap_depth(verts[i], verts[j], norms[i], norms[j])
            if d < -tol:
                return False
                
    # Check containment
    ocx, ocy = outer_center
    all_pts = np.vstack(verts)
    all_pts[:, 0] -= ocx
    all_pts[:, 1] -= ocy
    
    base_angles = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(outer_angle)
    normals = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    proj = all_pts @ normals.T
    max_proj = np.max(np.abs(proj))
    inradius = outer_side * np.sqrt(3.0) / 2.0
    
    if max_proj > inradius + tol:
        return False
        
    return True

def _objective(params, t0, deadline, beta=100.0):
    """
    Objective function for optimization.
    params: [x1, y1, r1, ..., x12, y12, r12, ox, oy, oang]
    Returns: Outer side length + penalty for overlaps.
    """
    if time.time() - t0 > deadline:
        return 1e6
        
    inner = params[:36].reshape(12, 3)
    ox = params[36]
    oy = params[37]
    oang = params[38]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
        
    # Calculate overlaps
    overlaps = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap_depth(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                overlaps.append(-d)
                
    if overlaps:
        p_arr = np.array(overlaps)
        # Smooth penalty: LogSumExp
        scaled = 3.0 * p_arr / beta
        max_s = np.max(scaled)
        if max_s > 500:
            # Fallback to quadratic for stability if overlaps are huge
            penalty = np.sum(p_arr**2)
        else:
            lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
            penalty = beta * lse
    else:
        penalty = 0.0

    # Calculate outer side
    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side(all_pts, oang)
    
    return L + 100.0 * penalty

def _generate_initial_config(type_id):
    """
    Generate initial configurations of different topologies.
    """
    sqrt3 = np.sqrt(3)
    
    if type_id == 0:
        # Compact Cluster (Hexagonal Lattice)
        # 3-4-3-2 or similar
        # Let's use a standard compact cluster of 12
        # Center at (0,0)
        # 1 center
        # 6 around center (distance sqrt(3))
        # 5 more around that?
        # Total 12.
        # Let's just use a rectangular-ish grid of hexagons
        # 4x3 grid
        pts = []
        for r in range(3):
            for c in range(4):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2
                pts.append([x, y, 0.0])
        
    elif type_id == 1:
        # Ring
        # 1 center, 6 inner ring, 5 outer ring?
        # Or 2 concentric rings
        pts = []
        pts.append([0, 0, 0.0])
        d1 = sqrt3
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        # Need 5 more
        d2 = 2 * sqrt3
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
            
    elif type_id == 2:
        # 2x6 Rectangle
        pts = []
        for r in range(2):
            for c in range(6):
                x = c * sqrt3
                y = r * 1.5
                pts.append([x, y, 0.0])
                
    else:
        # Random
        pts = []
        for _ in range(12):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            r = np.random.uniform(0, 60)
            pts.append([x, y, r])
            
    pts = np.array(pts)
    # Center the cluster
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    
    # Flatten
    params = np.zeros(39)
    params[:36] = pts.flatten()
    params[36] = 0.0 # Outer center x
    params[37] = 0.0 # Outer center y
    params[38] = 0.0 # Outer angle
    
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_side = float('inf')
    best_params = None
    best_valid = False
    
    # Try multiple initial configurations
    for init_type in range(3):
        if time.time() > deadline - 20:
            break
            
        p0 = _generate_initial_config(init_type)
        
        # Phase 1: L-BFGS-B with moderate penalty
        try:
            res1 = minimize(
                _objective,
                p0,
                args=(start_time, deadline, 50.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
            )
            if time.time() > deadline - 10:
                # Just check current best
                if res1.fun < best_side:
                    # Verify validity
                    inner = res1.x[:36].reshape(12, 3)
                    ox, oy, oang = res1.x[36], res1.x[37], res1.x[38]
                    side = _outer_hex_side(np.vstack([_hex_vertices(*v) for v in inner]) - [ox, oy], oang)
                    if _check_validity(inner, (ox, oy), side, oang):
                        best_side = side
                        best_params = res1.x
                        best_valid = True
                continue
                
            p1 = res1.x
            
            # Phase 2: L-BFGS-B with high penalty to enforce disjointness
            res2 = minimize(
                _objective,
                p1,
                args=(start_time, deadline, 200.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
            )
            
            p2 = res2.x
            
            # Phase 3: Nelder-Mead for final polish
            # Use a slightly different objective or just the same one
            # NM doesn't use gradients, so it can handle the kink better
            res3 = minimize(
                _objective,
                p2,
                args=(start_time, deadline, 500.0),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            p_final = res3.x
            
            # Extract and validate
            inner = p_final[:36].reshape(12, 3)
            ox, oy, oang = p_final[36], p_final[37], p_final[38]
            
            # Calculate exact side
            verts_list = [_hex_vertices(*v) for v in inner]
            all_pts = np.vstack(verts_list)
            all_pts[:, 0] -= ox
            all_pts[:, 1] -= oy
            side = _outer_hex_side(all_pts, oang)
            
            if _check_validity(inner, (ox, oy), side, oang):
                if side < best_side:
                    best_side = side
                    best_params = p_final
                    best_valid = True
                    
        except Exception as e:
            pass # Continue to next init if optimization fails

    # If no valid solution found, return a default valid one (e.g., large hexagon)
    if not best_valid or best_params is None:
        # Fallback: Large hexagon containing a simple grid
        p0 = _generate_initial_config(0)
        inner = p0[:36].reshape(12, 3)
        ox, oy, oang = 0.0, 0.0, 0.0
        verts_list = [_hex_vertices(*v) for v in inner]
        all_pts = np.vstack(verts_list)
        side = _outer_hex_side(all_pts, oang)
        best_params = p0
        best_side = side
        best_valid = True # It might be valid if we just scale up, but we didn't. 
        # Actually, if invalid, we need to ensure validity.
        # Let's just expand the outer hexagon until valid.
        while not _check_validity(inner, (ox, oy), best_side, oang):
            best_side *= 1.1
            if time.time() > deadline:
                break

    # Extract final result
    bx = best_params
    inner_x = bx[0:12]
    inner_y = bx[12:24]
    inner_angles = bx[24:36]
    ocx, ocy = bx[36], bx[37]
    oa = bx[38]
    
    inner = [[float(inner_x[i]), float(inner_y[i]), float(inner_angles[i])] for i in range(12)]
    
    return inner, (float(ocx), float(ocy)), float(best_side), float(oa)

# EVOLVE-BLOCK-END
