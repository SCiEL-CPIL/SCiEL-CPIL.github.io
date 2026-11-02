"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + np.deg2rad(60 * i)
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def point_to_hexagon_distance(px, py, hx, hy, hside, hangle):
    """
    Compute the minimum distance from point (px, py) to the boundary of a regular hexagon
    centered at (hx, hy) with side length hside and rotation hangle (degrees).
    If the point is outside, returns positive distance. If inside, returns negative distance.
    We use a smooth approximation via the minimum over edges.
    """
    verts = hex_vertices(hx, hy, hside, hangle)
    min_dist_sq = np.inf
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        # Distance from point to line segment v1-v2
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            dist_sq = (px - v1[0])**2 + (py - v1[1])**2
        else:
            t = ((px - v1[0]) * dx + (py - v1[1]) * dy) / length_sq
            t = max(0.0, min(1.0, t))
            proj_x = v1[0] + t * dx
            proj_y = v1[1] + t * dy
            dist_sq = (px - proj_x)**2 + (py - proj_y)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    
    min_dist = np.sqrt(min_dist_sq)
    
    # Determine if point is inside the hexagon
    # Use winding number or cross product method
    inside = True
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        cross = (v2[0] - v1[0]) * (py - v1[1]) - (v2[1] - v1[1]) * (px - v1[0])
        # For a convex polygon with consistent orientation, all crosses should have same sign
        # But we need to be careful. Let's use a different approach.
        pass
    
    # Simpler: check if point is inside using barycentric-like test for convex polygon
    # For a convex polygon, point is inside if it's on the same side of all edges
    signs = []
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        cross = (v2[0] - v1[0]) * (py - v1[1]) - (v2[1] - v1[1]) * (px - v1[0])
        signs.append(cross)
    
    # All crosses should have the same sign (or be zero) for point to be inside
    positive = any(s > 1e-10 for s in signs)
    negative = any(s < -1e-10 for s in signs)
    if positive and negative:
        inside = False
    else:
        inside = True
    
    if inside:
        return -min_dist
    else:
        return min_dist

def clearance_between_hexagons(h1, h2):
    """
    Compute the minimum clearance between two regular hexagons.
    h1 = (cx1, cy1, angle1), h2 = (cx2, cy2, angle2)
    Side length is 1 for both.
    Returns positive if disjoint, negative if overlapping.
    """
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    verts1 = hex_vertices(cx1, cy1, 1.0, a1)
    verts2 = hex_vertices(cx2, cy2, 1.0, a2)
    
    min_clearance = np.inf
    
    # Check each vertex of hex1 against each edge of hex2
    for v in verts1:
        for i in range(6):
            v1 = verts2[i]
            v2 = verts2[(i + 1) % 6]
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-12:
                dist_sq = (v[0] - v1[0])**2 + (v[1] - v1[1])**2
            else:
                t = ((v[0] - v1[0]) * dx + (v[1] - v1[1]) * dy) / length_sq
                t = max(0.0, min(1.0, t))
                proj_x = v1[0] + t * dx
                proj_y = v1[1] + t * dy
                dist_sq = (v[0] - proj_x)**2 + (v[1] - proj_y)**2
            dist = np.sqrt(dist_sq)
            if dist < min_clearance:
                min_clearance = dist
    
    # Check each vertex of hex2 against each edge of hex1
    for v in verts2:
        for i in range(6):
            v1 = verts1[i]
            v2 = verts1[(i + 1) % 6]
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-12:
                dist_sq = (v[0] - v1[0])**2 + (v[1] - v1[1])**2
            else:
                t = ((v[0] - v1[0]) * dx + (v[1] - v1[1]) * dy) / length_sq
                t = max(0.0, min(1.0, t))
                proj_x = v1[0] + t * dx
                proj_y = v1[1] + t * dy
                dist_sq = (v[0] - proj_x)**2 + (v[1] - proj_y)**2
            dist = np.sqrt(dist_sq)
            if dist < min_clearance:
                min_clearance = dist
    
    # Now determine if the hexagons are overlapping
    # A hexagon is inside another if all its vertices are inside
    # Or they overlap if some vertices of one are inside the other
    # For disjointness, we need no intersection of boundaries and no containment
    
    # Check if any vertex of hex1 is inside hex2
    for v in verts1:
        # Check if v is inside hex2
        signs = []
        for i in range(6):
            v1 = verts2[i]
            v2 = verts2[(i + 1) % 6]
            cross = (v2[0] - v1[0]) * (v[1] - v1[1]) - (v2[1] - v1[1]) * (v[0] - v1[0])
            signs.append(cross)
        positive = any(s > 1e-10 for s in signs)
        negative = any(s < -1e-10 for s in signs)
        if not (positive and negative):
            # v is inside hex2
            return -min_clearance if min_clearance > 0 else -0.1
    
    # Check if any vertex of hex2 is inside hex1
    for v in verts2:
        signs = []
        for i in range(6):
            v1 = verts1[i]
            v2 = verts1[(i + 1) % 6]
            cross = (v2[0] - v1[0]) * (v[1] - v1[1]) - (v2[1] - v1[1]) * (v[0] - v1[0])
            signs.append(cross)
        positive = any(s > 1e-10 for s in signs)
        negative = any(s < -1e-10 for s in signs)
        if not (positive and negative):
            # v is inside hex1
            return -min_clearance if min_clearance > 0 else -0.1
    
    return min_clearance

def outer_hexagon_clearance(inner_pos, outer_center, outer_side, outer_angle):
    """
    Compute the minimum clearance between all inner hexagon vertices and the outer hexagon boundary.
    Returns negative if any inner hexagon vertex is outside the outer hexagon.
    """
    ox, oy = outer_center
    outer_verts = hex_vertices(ox, oy, outer_side, outer_angle)
    
    min_clearance = np.inf
    
    for (ix, iy, iangle) in inner_pos:
        inner_verts = hex_vertices(ix, iy, 1.0, iangle)
        for v in inner_verts:
            # Distance from v to outer hexagon boundary
            min_dist = np.inf
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % 6]
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                length_sq = dx * dx + dy * dy
                if length_sq < 1e-12:
                    dist_sq = (v[0] - v1[0])**2 + (v[1] - v1[1])**2
                else:
                    t = ((v[0] - v1[0]) * dx + (v[1] - v1[1]) * dy) / length_sq
                    t = max(0.0, min(1.0, t))
                    proj_x = v1[0] + t * dx
                    proj_y = v1[1] + t * dy
                    dist_sq = (v[0] - proj_x)**2 + (v[1] - proj_y)**2
                dist = np.sqrt(dist_sq)
                if dist < min_dist:
                    min_dist = dist
            
            # Check if v is inside the outer hexagon
            signs = []
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % 6]
                cross = (v2[0] - v1[0]) * (v[1] - v1[1]) - (v2[1] - v1[1]) * (v[0] - v1[0])
                signs.append(cross)
            positive = any(s > 1e-10 for s in signs)
            negative = any(s < -1e-10 for s in signs)
            
            if positive and negative:
                # v is outside
                return -min_dist
            else:
                # v is inside
                if min_dist < min_clearance:
                    min_clearance = min_dist
    
    return min_clearance

def objective(params, N_inner):
    """
    Objective function to minimize. We want to minimize the outer side length.
    params: [ox, oy, outer_angle, outer_side, x1, y1, a1, x2, y2, a2, ...]
    We maximize the minimum clearance, which is equivalent to minimizing a penalty.
    """
    ox = params[0]
    oy = params[1]
    outer_angle = params[2]
    outer_side = params[3]
    
    inner_params = params[4:]
    inner_pos = []
    for i in range(N_inner):
        x = inner_params[i * 3]
        y = inner_params[i * 3 + 1]
        a = inner_params[i * 3 + 2]
        inner_pos.append((x, y, a))
    
    # Compute clearances between all pairs of inner hexagons
    min_inner_clearance = np.inf
    for i in range(N_inner):
        for j in range(i + 1, N_inner):
            c = clearance_between_hexagons(inner_pos[i], inner_pos[j])
            if c < min_inner_clearance:
                min_inner_clearance = c
    
    # Compute clearance to outer boundary
    outer_clearance = outer_hexagon_clearance(inner_pos, (ox, oy), outer_side, outer_angle)
    
    # We want all clearances to be positive
    # Penalty if any clearance is negative
    penalty = 0.0
    if min_inner_clearance < 0:
        penalty += 100.0 * (-min_inner_clearance)
    if outer_clearance < 0:
        penalty += 100.0 * (-outer_clearance)
    
    # We want to minimize outer_side, but also ensure feasibility
    # Use a weighted objective
    return outer_side + penalty

def construct():
    N_inner = N
    np.random.seed(42)
    
    # Initial guess: place hexagons in a grid-like pattern
    # For N=11, we can try a hexagonal arrangement
    # A hexagonal number arrangement: 1, 7, 19, ... 
    # 11 is not a hexagonal number, so we'll use a rectangular grid
    
    # Let's try a 3x4 grid with one missing, or similar
    # Actually, let's start with a simple grid and optimize
    
    # Initial positions: roughly in a circle
    positions = []
    for i in range(N_inner):
        angle = 2 * np.pi * i / N_inner
        r = 2.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        a = 0.0
        positions.append([x, y, a])
    
    # Build initial params
    init_params = [0.0, 0.0, 0.0, 15.0]  # ox, oy, outer_angle, outer_side
    for p in positions:
        init_params.extend(p)
    
    # Bounds
    bounds = []
    bounds.append((-50, 50))  # ox
    bounds.append((-50, 50))  # oy
    bounds.append((-180, 180))  # outer_angle
    bounds.append((3.0, 50.0))  # outer_side
    
    for i in range(N_inner):
        bounds.append((-30, 30))  # x
        bounds.append((-30, 30))  # y
        bounds.append((-180, 180))  # angle
    
    # Multi-start optimization
    best_result = None
    best_obj = np.inf
    
    n_starts = 5
    for start in range(n_starts):
        if start == 0:
            x0 = init_params
        else:
            # Perturb the initial guess
            x0 = init_params.copy()
            for i in range(4, len(x0)):
                if i % 3 == 0:
                    x0[i] += np.random.uniform(-2, 2)
                elif i % 3 == 1:
                    x0[i] += np.random.uniform(-2, 2)
                else:
                    x0[i] += np.random.uniform(-30, 30)
            x0[3] = 15.0 + np.random.uniform(-2, 2)
        
        try:
            result = minimize(objective, x0, args=(N_inner,), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success or result.fun < best_obj:
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback to initial
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    ox = best_result.x[0]
    oy = best_result.x[1]
    outer_angle = best_result.x[2]
    outer_side = best_result.x[3]
    
    inner = []
    for i in range(N_inner):
        x = best_result.x[4 + i * 3]
        y = best_result.x[5 + i * 3]
        a = best_result.x[6 + i * 3]
        inner.append([x, y, a])
    
    # Verify feasibility
    # Check all inner pairs
    for i in range(N_inner):
        for j in range(i + 1, N_inner):
            c = clearance_between_hexagons(inner[i], inner[j])
            if c < -1e-6:
                # Overlapping, fallback
                return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    
    # Check outer boundary
    oc = outer_hexagon_clearance(inner, (ox, oy), outer_side, outer_angle)
    if oc < -1e-6:
        # Some inner hexagons outside, fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    
    return inner, (ox, oy), outer_side, outer_angle

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
