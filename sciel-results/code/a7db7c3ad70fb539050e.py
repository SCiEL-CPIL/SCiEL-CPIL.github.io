import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 135.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * np.pi / 3
        verts[i, 0] = cx + s * np.cos(theta)
        verts[i, 1] = cy + s * np.sin(theta)
    return verts

def hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon with vertices at angle_deg + i*pi/3."""
    a = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = a + np.pi / 6.0 + i * np.pi / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) side s rotated angle_deg."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    for i in range(6):
        theta = i * np.pi / 3
        if lx * np.cos(theta) + ly * np.sin(theta) > s * np.sqrt(3) / 2 + 1e-12:
            return False
    return True

def hexagons_disjoint(v1, v2, s, eps=1e-9):
    """Check if two hexagons (vertex arrays) are disjoint (no interior intersection)."""
    def point_in_poly(px, py, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    for v in v1:
        if point_in_poly(v[0], v[1], v2):
            return False
    for v in v2:
        if point_in_poly(v[0], v[1], v1):
            return False
    def seg_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
    
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if seg_intersect(p1, p2, p3, p4):
                return False
    return True

def min_enclosing_hex_side(all_points, outer_angle_deg):
    """
    Compute minimum side length and center of a regular hexagon (fixed orientation)
    enclosing all points. Uses the fact that for a hexagon with center (cx,cy), side s,
    rotation a, the constraint is: for all face normals n_i,
    max over points of (p - c) . n_i <= s * sqrt(3)/2
    and min over points of (p - c) . n_i >= -s * sqrt(3)/2
    
    This means: for each normal direction, the range of projections must be <= 2 * s * sqrt(3)/2 = s * sqrt(3)
    But we also need to find the optimal center.
    
    For a given orientation, the minimal enclosing hexagon can be found by:
    - The center must be such that the max projection in each normal direction minus the min projection
      is minimized, but actually we need the max of (max_proj_i - (-min_proj_i)) / sqrt(3) over all i.
    
    Actually, let's use a simpler approach: sweep the center.
    For a fixed orientation, the optimal center is the centroid of the "critical" points.
    
    Let's use a grid search or optimization.
    """
    sqrt3 = np.sqrt(3)
    normals = hex_edge_normals(outer_angle_deg)
    
    # For each normal direction i, let:
    #   max_i = max over points of (p . n_i)
    #   min_i = min over points of (p . n_i)
    # The constraint for center c is:
    #   max_i - c . n_i <= s * sqrt3/2
    #   -min_i + c . n_i <= s * sqrt3/2
    # So: c . n_i <= max_i - s*sqrt3/2
    #     c . n_i >= min_i + s*sqrt3/2
    # For feasibility: min_i + s*sqrt3/2 <= max_i - s*sqrt3/2
    #   => s >= (max_i - min_i) / sqrt3
    # And the center must satisfy all 12 half-plane constraints.
    
    # The minimal s is max over i of (max_i - min_i) / sqrt3
    # But we also need to check if a center exists.
    
    max_projs = np.array([all_points @ normals[i] for i in range(6)]).max(axis=1)
    min_projs = np.array([all_points @ normals[i] for i in range(6)]).min(axis=1)
    
    # Minimal side from range constraint
    s_min = np.max(max_projs - min_projs) / sqrt3
    
    # Now find a center that works with s = s_min
    # Center c must satisfy:
    #   c . n_i <= max_i - s*sqrt3/2  for all i
    #   c . n_i >= min_i + s*sqrt3/2  for all i
    # This is a system of linear inequalities. Let's use a simple approach:
    # The feasible region is a convex polygon (intersection of 12 half-planes).
    # We can find a point in it by checking if the centroid of the "tight" constraints works.
    
    # For now, let's use the midpoint of the bounding box in each normal direction
    # and verify. If it doesn't work, we'll increase s slightly.
    
    s = s_min
    for _ in range(10):
        # Try center as average of midpoints
        midpoints = (max_projs + min_projs) / 2
        # This isn't directly a point in 2D, so let's use a different approach
        
        # Use scipy to solve the feasibility problem
        from scipy.optimize import linprog
        # Minimize 0 subject to:
        #   n_i . c <= max_i - s*sqrt3/2
        #   -n_i . c <= -(min_i + s*sqrt3/2)
        A_ub = np.vstack([normals, -normals])
        b_ub = np.concatenate([max_projs - s * sqrt3 / 2, -(min_projs + s * sqrt3 / 2)])
        
        res = linprog(np.zeros(2), A_ub=A_ub, b_ub=b_ub, method='highs')
        if res.status == 0:
            return s, (res.x[0], res.x[1])
        else:
            # Increase s slightly
            s *= 1.001
    
    # Fallback: use centroid and compute required s
    cx = all_points[:, 0].mean()
    cy = all_points[:, 1].mean()
    max_viol = 0
    for i in range(6):
        proj = (all_points[:, 0] - cx) * normals[i, 0] + (all_points[:, 1] - cy) * normals[i, 1]
        max_viol = max(max_viol, proj.max(), -proj.min())
    s = max_viol / (sqrt3 / 2)
    return s, (cx, cy)

def optimize_outer_hex(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    sqrt3 = np.sqrt(3)
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    all_points = np.vstack([hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # Sweep outer angle finely
    for outer_angle in np.linspace(0, 60, 121):
        side, center = min_enclosing_hex_side(all_points, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle

def hex_overlap_sat(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    v1 = hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2
    
    for i in range(len(centers)):
        verts = hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(N):
        for j in range(i + 1, N):
            if hex_overlap_sat(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True

def construct():
    start_time = time.time()
    
    sqrt3 = np.sqrt(3)
    
    # Initial hexagonal cluster arrangement
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    centers = np.array(centers)
    inner_angles = [0.0] * N
    
    # Verify initial arrangement is valid
    inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Scale up slightly to ensure disjointness
        scale = 1.01
        centers = centers * scale
        inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                    valid = False
                    break
            if not valid:
                break
    
    # Get initial outer hexagon
    best_side, best_center, best_angle = optimize_outer_hex(centers, inner_angles)
    
    # Try to improve using force-directed relaxation
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        # Define objective: minimize outer hexagon side length
        # Variables: flattened [x0,y0,x1,y1,...,x10,y10]
        def objective(vars):
            if time.time() - start_time > DEADLINE:
                return 1e10
            c = np.array(vars).reshape(N, 2)
            # Check overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    if hex_overlap_sat((c[i][0], c[i][1]), 0.0, (c[j][0], c[j][1]), 0.0):
                        return 1e10
            side, _, _ = optimize_outer_hex(c, inner_angles)
            return side
        
        x0 = centers.flatten()
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_side and time.time() - start_time < DEADLINE:
            new_centers = result.x.reshape(N, 2)
            # Verify
            new_inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in new_centers]
            new_valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if not hexagons_disjoint(new_inner_verts[i], new_inner_verts[j], 1.0):
                        new_valid = False
                        break
                if not new_valid:
                    break
            if new_valid:
                centers = new_centers
                best_side, best_center, best_angle = optimize_outer_hex(centers, inner_angles)
    
    # Build inner list
    inner = [[float(cx), float(cy), float(0.0)] for cx, cy in centers]
    
    # Final verification and safety margin
    best_side *= 1.001
    inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
    all_check_verts = np.vstack(inner_verts)
    
    a = np.radians(best_angle)
    normals = hex_edge_normals(best_angle)
    
    dx = all_check_verts[:, 0] - best_center[0]
    dy = all_check_verts[:, 1] - best_center[1]
    
    max_violation = 0
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        max_violation = max(max_violation, proj.max() - best_side * sqrt3 / 2)
        max_violation = max(max_violation, -proj.min() - best_side * sqrt3 / 2)
    
    if max_violation > 1e-9:
        best_side += max_violation / (sqrt3 / 2) * 1.001
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
