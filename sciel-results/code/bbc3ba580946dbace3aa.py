"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    Uses log-sum-exp for smooth max projection to aid gradient-based optimization.
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
    
    projections = []
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
        # projections shape: (6, 6) -> 6 vertices x 6 normals
        projs = rotated @ normals.T
        
        # We need the max over all vertices and normals? 
        # No, the condition is that ALL vertices must be inside.
        # A point is inside a hexagon if its projection onto every normal is <= S * sqrt(3)/2.
        # So we need the max projection over ALL vertices and ALL normals.
        projections.extend(projs.flatten())
    
    if not projections:
        return 0.0
    
    projections = np.array(projections)
    
    # Log-sum-exp for smooth max
    max_proj = np.max(projections)
    if max_proj < -1000:
        return 0.0
    
    # Smooth max via log-sum-exp: (1/beta) * log(sum(exp(beta * p_i)))
    scaled = beta * (projections - max_proj)
    sum_exp = np.sum(np.exp(scaled))
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * smooth_max / SQRT3

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
        # Edge i is from v[i] to v[i+1].
        # Normal points outward.
        # For a regular hexagon, outward normal at vertex i (angle theta_i) is at theta_i + 30 deg?
        # Vertices are at 0, 60, ...
        # Edge 0 is between v0 and v1. Midpoint angle 30. Normal is 30.
        # So normals are at 30, 90, 150, 210, 270, 330 + rotation.
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
            # projs1 shape (12, 6), projs2 shape (12, 6)
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
            # D = max(d)
            max_d = np.max(d)
            scaled = beta * (d - max_d)
            sum_exp = np.sum(np.exp(scaled))
            D_smooth = max_d + math.log(sum_exp) / beta
            
            # Penalty if D_smooth is negative (overlap)
            # softplus(-D_smooth)
            # softplus(x) = log(1 + exp(x))
            # We want to penalize when D_smooth < 0.
            # Let z = -D_smooth. If D_smooth < 0, z > 0.
            # Penalty = softplus(z) = log(1 + exp(z)).
            # This is smooth and > 0.
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
    
    # Use smooth side length calculation
    S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=50.0)
    
    # Add smooth overlap penalty
    penalty = compute_overlap_penalty(inner, beta=50.0)
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice - use tight packing
    # Axial coordinates for hexagonal lattice
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,-1), (1,1), (-2,1), (-1,-1), (0,2), (2,-2) ... pick 4 compact ones
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian with tight packing
    # To ensure initial non-overlap, we scale up slightly by epsilon
    epsilon = 1e-6
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    # This is a valid non-overlapping configuration.
    initial_inner = [[cx, cy, 0.0] for cx, cy in centers]
    
    # Initial outer hexagon
    # Compute required side for this initial layout
    init_outer_center = (0.0, 0.0)
    init_outer_angle = 0.0
    init_S = compute_required_side_smooth(initial_inner, init_outer_center, init_outer_angle, beta=100.0)
    
    # Build initial params vector
    params0 = np.zeros(3 * N + 3)
    for i, (cx, cy, ang) in enumerate(initial_inner):
        params0[3*i] = cx
        params0[3*i+1] = cy
        params0[3*i+2] = np.radians(ang)
    params0[3*N] = init_outer_center[0]
    params0[3*N+1] = init_outer_center[1]
    params0[3*N+2] = np.radians(init_outer_angle)
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple initial conditions / optimizations
    initial_conditions = [params0]
    
    # Add perturbed versions
    np.random.seed(42)
    for _ in range(3):
        perturbed = params0 + np.random.randn(3*N+3) * 0.1
        # Keep angles small
        for i in range(N):
            perturbed[3*i+2] = np.random.randn() * 0.2
        initial_conditions.append(perturbed)
    
    for ic in initial_conditions:
        if time.time() > deadline - 40:
            break
        
        try:
            res = minimize(objective, ic, args=(deadline,), method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000})
            
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
                
                # Compute exact S for validity check
                S_exact = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=1000.0) # High beta approximates max
                
                if check_validity(inner, (ox, oy), S_exact, outer_angle):
                    if S_exact < best_score:
                        best_score = S_exact
                        best_result = (inner, (ox, oy), S_exact, outer_angle)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        S_fallback = compute_required_side_smooth(initial_inner, (0.0, 0.0), 0.0, beta=1000.0)
        best_result = (initial_inner, (0.0, 0.0), S_fallback, 0.0)
        best_score = S_fallback
    
    # Refine outer hexagon parameters using Nelder-Mead
    if time.time() < deadline - 20:
        try:
            inner_fixed = best_result[0]
            
            def outer_obj(params_outer):
                ox, oy, oa = params_outer
                S = compute_required_side_smooth(inner_fixed, (ox, oy), np.degrees(oa), beta=1000.0)
                return S
            
            outer_ic = np.array([best_result[1][0], best_result[1][1], np.radians(best_result[3])])
            res_outer = minimize(outer_obj, outer_ic, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_outer.fun < best_score:
                ox, oy, oa = res_outer.x
                outer_angle = np.degrees(oa) % 360
                S = res_outer.fun
                if check_validity(inner_fixed, (ox, oy), S, outer_angle):
                    best_result = (inner_fixed, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Final refinement of all parameters with tighter tolerance
    if time.time() < deadline - 10:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res = minimize(objective, params_best, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            
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
                S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=1000.0)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
