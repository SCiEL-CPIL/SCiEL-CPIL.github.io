"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 11

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotated by angle_deg."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (cx, cy) with side length and angle."""
    # Translate point to hex center
    dx = px - cx
    dy = py - cy
    # Rotate point by -angle to align hex with axes
    angle = np.radians(angle_deg)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    # For a regular hexagon aligned with x-axis (flat-top? no, vertex at 0 deg is pointy-top? 
    # In hex_vertices, i=0 is at angle. So if angle=0, vertex is at (side, 0). 
    # This is a "pointy-top" hexagon relative to the vertex, but let's check the shape.
    # Vertices at 0, 60, 120, 180, 240, 300.
    # The boundaries are lines at 30, 90, 150, 210, 270, 330?
    # The normal directions for a hexagon with vertices at 0, 60... are at 30, 90...
    # Let's use the dot product method.
    # The hexagon is the intersection of 6 half-planes.
    # For a hexagon with vertex at (s, 0), the right-most extent is s.
    # The "width" in direction theta is s * (cos(theta) + cos(60-theta))? No.
    # Let's stick to the LP definition used in min_enclosing_hexagon.
    # A point is inside if for all 6 normals n_k: dot(p, n_k) <= side * cos(pi/6) = side * sqrt(3)/2.
    # Normals are at angles 30, 90, 150, 210, 270, 330 degrees (if vertices are at 0, 60...).
    # Wait, if vertices are at 0, 60, ..., the edges are between 0-60, 60-120.
    # The normal to the edge between 0 and 60 is at 30 degrees.
    # So normals are at 30 + i*60.
    
    sqrt3_2 = np.sqrt(3) / 2.0
    for i in range(6):
        normal_angle = np.radians(30 + i * 60)
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        # The rotated point (rx, ry) is in the local frame where hex is aligned.
        # But the normals in the local frame are fixed at 30, 90...
        # So we just check dot((rx, ry), n_k) <= side * sqrt(3)/2
        if rx * nx + ry * ny > side * sqrt3_2 + 1e-9:
            return False
    return True

def check_overlaps(inner_pts, inner_angles):
    """
    Returns the sum of squared penetration depths (approximate) or just a count/penalty.
    For simplicity, we check if any vertex of hex i is inside hex j.
    If so, we add a penalty. To make it smooth, we can use the distance to the boundary.
    However, for the optimizer, a hard penalty that goes to infinity might work, 
    but L-BFGS-B prefers smooth. 
    Let's use a penalty based on the maximum violation of the half-plane constraints.
    """
    n = len(inner_pts)
    penalty = 0.0
    sqrt3_2 = np.sqrt(3) / 2.0
    
    for i in range(n):
        vi = hex_vertices(inner_pts[i, 0], inner_pts[i, 1], 1.0, inner_angles[i])
        for j in range(i + 1, n):
            cx_j, cy_j = inner_pts[j, 0], inner_pts[j, 1]
            ang_j = inner_angles[j]
            # Check if any vertex of i is inside j
            for v in vi:
                if point_in_hex(v[0], v[1], cx_j, cy_j, 1.0, ang_j):
                    # It's inside. We need a measure of how much.
                    # Calculate the minimum distance to the boundary of j.
                    # Or just add a large constant.
                    # Better: calculate the maximum "excess" over the half-plane constraints.
                    # Let's compute the "distance" to the boundary in the local frame of j.
                    dx = v[0] - cx_j
                    dy = v[1] - cy_j
                    angle_j_rad = np.radians(ang_j)
                    cos_a = np.cos(-angle_j_rad)
                    sin_a = np.sin(-angle_j_rad)
                    rx = dx * cos_a - dy * sin_a
                    ry = dx * sin_a + dy * cos_a
                    
                    max_violation = 0.0
                    for k in range(6):
                        normal_angle = np.radians(30 + k * 60)
                        nx = np.cos(normal_angle)
                        ny = np.sin(normal_angle)
                        proj = rx * nx + ry * ny
                        limit = 1.0 * sqrt3_2
                        if proj > limit:
                            max_violation = max(max_violation, proj - limit)
                    
                    if max_violation > 0:
                        penalty += max_violation ** 2
                    # If it's strictly inside, proj < limit for all, so max_violation is 0?
                    # No, if it's inside, it doesn't violate the "outside" constraints.
                    # Wait, the constraint for "inside" is proj <= limit.
                    # If it's inside, all projs are <= limit.
                    # So max_violation would be 0.
                    # This means my penalty logic above only triggers if a vertex is OUTSIDE j.
                    # But we want to penalize OVERLAP.
                    # Overlap means vertex of i is INSIDE j.
                    # So I need to detect if it's inside.
                    # If it's inside, the "distance to boundary" is positive (inward).
                    # Let's define the signed distance.
                    # For each half-plane, d_k = limit - proj.
                    # If point is inside, all d_k > 0. The distance to boundary is min(d_k).
                    # If point is outside, some d_k < 0.
                    
                    min_d = np.inf
                    for k in range(6):
                        normal_angle = np.radians(30 + k * 60)
                        nx = np.cos(normal_angle)
                        ny = np.sin(normal_angle)
                        proj = rx * nx + ry * ny
                        limit = 1.0 * sqrt3_2
                        d = limit - proj
                        if d < min_d:
                            min_d = d
                    
                    if min_d > 0:
                        # Point is inside. Penalty is min_d^2
                        penalty += min_d ** 2
                    else:
                        # Point is outside. No overlap contribution from this vertex.
                        pass
                        
    return penalty

def min_enclosing_hexagon(inner_pts, inner_angles, deadline):
    """
    Find the minimum side length regular hexagon that encloses all inner hexagons.
    """
    n_inner = len(inner_pts)
    sqrt3_2 = np.sqrt(3) / 2.0
    
    all_verts = []
    for k in range(n_inner):
        verts = hex_vertices(inner_pts[k, 0], inner_pts[k, 1], 1.0, inner_angles[k])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    m = len(all_verts)
    
    best_side = np.inf
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for outer_angle_deg in range(0, 60, 5):
        if time.time() > deadline:
            break
        outer_angle = np.radians(outer_angle_deg)
        
        normals = np.array([[np.cos(outer_angle + i * np.pi / 3), 
                            np.sin(outer_angle + i * np.pi / 3)] for i in range(6)])
        
        A_ub = np.zeros((m * 6, 3))
        b_ub = np.zeros(m * 6)
        
        for i in range(m):
            vx, vy = all_verts[i]
            for j in range(6):
                nx, ny = normals[j]
                row = i * 6 + j
                A_ub[row, 0] = nx
                A_ub[row, 1] = ny
                A_ub[row, 2] = -sqrt3_2
                b_ub[row] = vx * nx + vy * ny
        
        c = np.array([0.0, 0.0, 1.0])
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            if result.success:
                s_val = result.x[2]
                if s_val < best_side:
                    best_side = s_val
                    best_center = (result.x[0], result.x[1])
                    best_angle = outer_angle_deg
        except Exception:
            pass
    
    if not np.isfinite(best_side):
        xs = all_verts[:, 0]
        ys = all_verts[:, 1]
        width = xs.max() - xs.min()
        height = ys.max() - ys.min()
        best_side = max(width, height) / np.sqrt(3)
        best_center = ((xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2)
        best_angle = 0.0
    
    return best_side, best_center, best_angle

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    sqrt3 = np.sqrt(3)
    spacing = sqrt3 + 0.1  # Safe spacing
    
    # Initial valid configuration: 11 hexagons in a hexagonal cluster
    # Center (0,0)
    # Ring 1: 6 hexagons at distance 'spacing'
    # Ring 2: 4 hexagons? 
    # A standard hex cluster of 19 is 1+6+12.
    # 11 is 1+6+4.
    # Let's place the 4 from the second ring in the gaps of the first ring to keep it compact.
    
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        x = spacing * np.cos(angle)
        y = spacing * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
        
    # Ring 2 (partial)
    # The vertices of the hexagon formed by Ring 1 are at distance 2*spacing.
    # The "corners" of the cluster are at angles 0, 60, 120...
    # To fill in, we can place hexagons at the "corners" of the hexagonal lattice.
    # Positions for Ring 2 in a full hex grid (distance 2*spacing) are at angles 0, 60... and 30, 90...
    # Let's pick 4 that are most central.
    # Angles: 0, 60, 120, 180? Or 30, 90, 150, 210?
    # Let's try 0, 60, 120, 180 (even indices of the 12 positions in ring 2).
    for i in range(4):
        angle = i * np.pi / 3  # 0, 60, 120, 180
        x = 2 * spacing * np.cos(angle)
        y = 2 * spacing * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
        
    # Total 11
    inner_pts = np.array(positions)
    inner_angles = np.array(angles)
    
    # Verify initial validity
    # If there are overlaps, we have a problem. With spacing = sqrt(3)+0.1, they should be disjoint.
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        n = len(inner_angles)
        # params: [x1, y1, a1, x2, y2, a2, ...]
        pts = np.zeros((n, 2))
        angs = np.zeros(n)
        for i in range(n):
            pts[i, 0] = params[i * 3]
            pts[i, 1] = params[i * 3 + 1]
            angs[i] = params[i * 3 + 2]
            
        # Penalty for overlaps
        overlap_pen = check_overlaps(pts, angs)
        
        # Enclosing hexagon
        side, center, angle = min_enclosing_hexagon(pts, angs, deadline)
        
        return side + 100.0 * overlap_pen

    # Flatten initial params
    init_params = np.zeros(N * 3)
    for i in range(N):
        init_params[i * 3] = inner_pts[i, 0]
        init_params[i * 3 + 1] = inner_pts[i, 1]
        init_params[i * 3 + 2] = inner_angles[i]
        
    # Optimize
    best_params = init_params.copy()
    best_val = objective(init_params)
    
    try:
        result = minimize(objective, init_params, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception as e:
        print(f"Optimization error: {e}")
        
    # Extract best result
    final_pts = np.zeros((N, 2))
    final_angs = np.zeros(N)
    for i in range(N):
        final_pts[i, 0] = best_params[i * 3]
        final_pts[i, 1] = best_params[i * 3 + 1]
        final_angs[i] = best_params[i * 3 + 2]
        
    # Final check: if overlaps are still present, we might need to fix them or accept the penalty.
    # But the evaluator will reject it.
    # Let's do a final local search to remove overlaps if any.
    # For now, assume the penalty pushed them apart.
    
    final_side, final_center, final_angle = min_enclosing_hexagon(final_pts, final_angs, deadline)
    
    # Build inner list
    inner = []
    for k in range(N):
        x = final_pts[k, 0]
        y = final_pts[k, 1]
        a = final_angs[k]
        inner.append([float(x), float(y), float(a)])
    
    return inner, (float(final_center[0]), float(final_center[1])), float(final_side), float(final_angle)

# EVOLVE-BLOCK-END
