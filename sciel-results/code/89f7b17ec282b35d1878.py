"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side_exact(inner, outer_center, outer_angle_deg):
    """
    Compute the exact minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = np.array([
        [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))] 
        for i in range(6)
    ])
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    # Rotation matrix for -outer_angle
    R = np.array([
        [cos_a, sin_a],
        [-sin_a, cos_a]
    ])
    
    max_proj = -float('inf')
    
    for (ix, iy, iangle) in inner:
        # Compute vertices of this inner hexagon
        theta0 = math.radians(iangle)
        angles = theta0 + np.pi/3 * np.arange(6)
        verts_x = ix + 1.0 * np.cos(angles)
        verts_y = iy + 1.0 * np.sin(angles)
        verts = np.column_stack((verts_x, verts_y))
        
        # Translate and rotate vertices to outer frame
        # v' = R * (v - center)
        centered = verts - np.array([ox, oy])
        rotated = centered @ R.T
        
        # Project onto all 6 normals
        projs = rotated @ normals.T
        
        # We need the max projection over ALL vertices and ALL normals.
        # A point is inside a hexagon if its projection onto every normal is <= S * sqrt(3)/2.
        # So we need the max projection over ALL vertices and ALL normals.
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
            
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_list = [get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                px = vx - p1[0]
                py = vy - p1[1]
                cross = ex * py - ey * px
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def solve_lp_for_layout(pts, beta_rad):
    """
    Solves the LP for a given set of points (centers) and outer rotation.
    Finds optimal center and minimum D.
    Returns (cx, cy, D) or (None, None, inf)
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return None, None, float('inf')
    
    # Normals of the outer hexagon
    normals = []
    for k in range(6):
        ang = beta_rad + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    # Variables: cx, cy, D
    # Minimize D
    # Constraints: | (p_i - c) . n_k | <= D
    # i.e., (p_i - c) . n_k <= D AND -(p_i - c) . n_k <= D
    # (p_i - c) . n_k - D <= 0  =>  -cx*nx - cy*ny - D <= -p_i.n_k
    # -(p_i - c) . n_k - D <= 0 =>  cx*nx + cy*ny - D <= p_i.n_k
    
    c_obj = np.array([0.0, 0.0, 1.0])
    
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in pts:
            # Constraint 1: (px - cx)nx + (py - cy)ny <= D
            # -cx*nx - cy*ny - D <= -px*nx - py*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            
            # Constraint 2: -(px - cx)nx - (py - cy)ny <= D
            # cx*nx + cy*ny - D <= px*nx + py*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            cx = res.x[0]
            cy = res.x[1]
            D = res.x[2]
            return cx, cy, D
        else:
            return None, None, float('inf')
    except Exception:
        return None, None, float('inf')

def optimize_rigid_composite(axial_coords, scale=1.0):
    """
    Treats all hexagons as a single rigid body (same relative positions, same orientation).
    Optimizes the orientation and position of the composite body to minimize the bounding hexagon size.
    """
    # Generate initial centers
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
    
    pts = np.array(centers)
    
    best_S = float('inf')
    best_beta = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Since the composite is rigid and symmetric, we only need to optimize the global rotation beta.
    # The optimal center (cx, cy) can be found via LP for any given beta.
    # Let's scan beta from 0 to 60 degrees (symmetry period is 60 deg for hexagons, but the shape might have lower symmetry).
    # Actually, the shape of 11 hexagons (center + 6 + 4) has 3-fold or 6-fold symmetry?
    # Center + 6 ring is 6-fold. Adding 4 from ring 2 breaks 6-fold to maybe 3-fold or less.
    # Let's just scan 0 to 60.
    
    # Use Nelder-Mead to optimize beta
    def obj_beta(beta_rad):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        cx, cy, D = solve_lp_for_layout(pts, beta_rad)
        if D == float('inf'):
            return float('inf')
        S = (D + APOTHEM) / APOTHEM
        return S
    
    # Initial guess
    beta0 = 0.0
    
    try:
        res = minimize(obj_beta, [beta0], method='Nelder-Mead',
                       options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_S:
            best_S = res.fun
            best_beta = res.x[0]
            
            # Get optimal center for this beta
            cx, cy, D = solve_lp_for_layout(pts, best_beta)
            if D != float('inf'):
                best_cx = cx
                best_cy = cy
    except Exception:
        pass
    
    # Also try a few discrete angles to be safe
    for beta_deg in [0, 15, 30, 45, 60]:
        beta_rad = math.radians(beta_deg)
        cx, cy, D = solve_lp_for_layout(pts, beta_rad)
        if D != float('inf'):
            S = (D + APOTHEM) / APOTHEM
            if S < best_S:
                best_S = S
                best_beta = beta_rad
                best_cx = cx
                best_cy = cy
                
    return best_S, best_cx, best_cy, best_beta

def compute_overlap_penalty(inner, beta=50.0):
    """
    Compute a smooth penalty for overlapping inner hexagons using SAT.
    """
    n = len(inner)
    penalty = 0.0
    
    # Precompute vertices and normals for all hexagons
    verts_list = []
    normals_list = []
    for i in range(n):
        cx, cy, ang = inner[i]
        theta0 = math.radians(ang)
        angles = theta0 + np.pi/3 * np.arange(6)
        verts = np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))
        verts_list.append(verts)
        
        # Normals are perpendicular to edges.
        normal_angles = theta0 + math.pi/6 + np.pi/3 * np.arange(6)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        normals_list.append(normals)

    for i in range(n):
        for j in range(i + 1, n):
            v1 = verts_list[i]
            v2 = verts_list[j]
            n1 = normals_list[i]
            n2 = normals_list[j]
            
            all_normals = np.vstack([n1, n2])
            
            # Project v1 and v2 onto all normals
            projs1 = v1 @ all_normals.T
            projs2 = v2 @ all_normals.T
            
            min1 = np.min(projs1, axis=1)
            max1 = np.max(projs1, axis=1)
            min2 = np.min(projs2, axis=1)
            max2 = np.max(projs2, axis=1)
            
            # d_n = min(max1 - min2, max2 - min1)
            d1 = max1 - min2
            d2 = max2 - min1
            d = np.minimum(d1, d2)
            
            # Smooth max of d
            max_d = np.max(d)
            scaled = beta * (d - max_d)
            sum_exp = np.sum(np.exp(scaled))
            D_smooth = max_d + math.log(sum_exp) / beta
            
            # Penalty if D_smooth is negative (overlap)
            z = -D_smooth
            # To avoid overflow in exp(z) if z is large, use stable softplus
            if z > 20:
                penalty += z
            elif z < -20:
                penalty += 0
            else:
                penalty += math.log(1 + math.exp(z))

    return penalty

def objective(params, deadline):
    """Objective function: outer side length + smooth penalty for overlaps."""
    if time.time() > deadline:
        return float('inf')
    
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    
    # Use exact side length calculation for better precision in the objective
    # The overlap penalty handles the disjointness constraint smoothly.
    S = compute_required_side_exact(inner, (ox, oy), outer_angle)
    
    # Add smooth overlap penalty
    penalty = compute_overlap_penalty(inner, beta=50.0)
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # 1. Define 11 hexagons on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Initial layout: Rigid composite optimization
    # We use a slight scale up to ensure strict disjointness for the LP phase, 
    # but we will refine with full DOF afterwards.
    # Actually, for the rigid phase, we can assume they are touching (scale=1.0) 
    # because the LP finds the minimum bounding box. If they touch, it's valid.
    # However, the subsequent optimization might break them apart slightly if it helps.
    
    # Phase 1: Rigid Composite Optimization
    # This reduces 36 DOF to 3 DOF (cx, cy, angle) effectively.
    S_rigid, cx_rigid, cy_rigid, angle_rigid = optimize_rigid_composite(axial_coords, scale=1.0)
    
    # Construct initial inner list based on rigid solution
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
        
    # Rotate and translate centers to match the rigid solution
    cos_a = math.cos(angle_rigid)
    sin_a = math.sin(angle_rigid)
    
    initial_inner = []
    for (lx, ly) in centers:
        # Rotate
        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a
        # Translate
        fx = rx + cx_rigid
        fy = ry + cy_rigid
        # All have the same global angle
        initial_inner.append([fx, fy, np.degrees(angle_rigid) % 360])
        
    # Build initial params vector
    params0 = np.zeros(3 * N + 3)
    for i, (cx, cy, ang) in enumerate(initial_inner):
        params0[3*i] = cx
        params0[3*i+1] = cy
        params0[3*i+2] = np.radians(ang)
    params0[3*N] = cx_rigid
    params0[3*N+1] = cy_rigid
    params0[3*N+2] = angle_rigid
    
    best_result = None
    best_score = float('inf')
    
    # Check validity of rigid solution
    S_exact_rigid = compute_required_side_exact(initial_inner, (cx_rigid, cy_rigid), np.degrees(angle_rigid) % 360)
    if check_validity(initial_inner, (cx_rigid, cy_rigid), S_exact_rigid, np.degrees(angle_rigid) % 360):
        best_score = S_exact_rigid
        best_result = (initial_inner, (cx_rigid, cy_rigid), S_exact_rigid, np.degrees(angle_rigid) % 360)
    else:
        # If rigid solution is invalid (shouldn't happen with scale=1.0 and touching allowed), 
        # try with a tiny scale up
        S_rigid, cx_rigid, cy_rigid, angle_rigid = optimize_rigid_composite(axial_coords, scale=1.0001)
        centers = []
        for q, r in axial_coords:
            x = 1.5 * 1.0001 * q
            y = SQRT3 * 1.0001 * (r + q / 2.0)
            centers.append((x, y))
            
        cos_a = math.cos(angle_rigid)
        sin_a = math.sin(angle_rigid)
        initial_inner = []
        for (lx, ly) in centers:
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            fx = rx + cx_rigid
            fy = ry + cy_rigid
            initial_inner.append([fx, fy, np.degrees(angle_rigid) % 360])
            
        params0 = np.zeros(3 * N + 3)
        for i, (cx, cy, ang) in enumerate(initial_inner):
            params0[3*i] = cx
            params0[3*i+1] = cy
            params0[3*i+2] = np.radians(ang)
        params0[3*N] = cx_rigid
        params0[3*N+1] = cy_rigid
        params0[3*N+2] = angle_rigid
        
        S_exact_rigid = compute_required_side_exact(initial_inner, (cx_rigid, cy_rigid), np.degrees(angle_rigid) % 360)
        if check_validity(initial_inner, (cx_rigid, cy_rigid), S_exact_rigid, np.degrees(angle_rigid) % 360):
            best_score = S_exact_rigid
            best_result = (initial_inner, (cx_rigid, cy_rigid), S_exact_rigid, np.degrees(angle_rigid) % 360)

    # Phase 2: Full DOF Refinement with L-BFGS-B
    # We allow individual hexagons to rotate and translate slightly to break the rigid constraint
    # and potentially find a tighter packing.
    
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, params0, args=(deadline,), method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                
                S_exact = compute_required_side_exact(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S_exact, outer_angle):
                    if S_exact < best_score:
                        best_score = S_exact
                        best_result = (inner, (ox, oy), S_exact, outer_angle)
        except Exception:
            pass
            
    # Phase 3: Polish with Nelder-Mead on outer parameters only
    if best_result is not None and time.time() < deadline - 10:
        try:
            inner_fixed = best_result[0]
            
            def outer_obj(params_outer):
                ox, oy, oa = params_outer
                S = compute_required_side_exact(inner_fixed, (ox, oy), np.degrees(oa))
                return S
            
            outer_ic = np.array([best_result[1][0], best_result[1][1], np.radians(best_result[3])])
            res_outer = minimize(outer_obj, outer_ic, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_outer.fun < best_score:
                ox, oy, oa = res_outer.x
                outer_angle = np.degrees(oa) % 360
                S = res_outer.fun
                if check_validity(inner_fixed, (ox, oy), S, outer_angle):
                    best_result = (inner_fixed, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
            
    # Fallback
    if best_result is None:
        # Simple lattice fallback
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        initial_inner = [[cx, cy, 0.0] for cx, cy in centers]
        S_fallback = compute_required_side_exact(initial_inner, (0.0, 0.0), 0.0)
        best_result = (initial_inner, (0.0, 0.0), S_fallback, 0.0)
        best_score = S_fallback
        
    return best_result

# EVOLVE-BLOCK-END
