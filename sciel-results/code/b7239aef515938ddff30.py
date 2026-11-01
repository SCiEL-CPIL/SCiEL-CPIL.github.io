"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons.
Combines LP-based exact objective with gradient-based local refinement."""
import numpy as np
import time
import math
from scipy.optimize import minimize, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def compute_required_side(inner, outer_center, outer_angle_deg):
    """Compute the minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_lp(inner, outer_center, outer_angle_deg):
    """
    Use LP to find the minimum enclosing hexagon.
    Returns (S, center, circumradius) or (inf, None, None) on failure.
    """
    if not inner:
        return 0.0, (0.0, 0.0), 0.0
    
    ox, oy = outer_center
    beta = math.radians(outer_angle_deg)
    
    # Normals for the 6 sides of the outer hexagon (pointing inward)
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    # Collect all vertex points from inner hexagons
    points = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            points.append((v[0], v[1]))
    
    if not points:
        return 0.0, (0.0, 0.0), 0.0
    
    # LP: minimize D (circumradius) subject to:
    # For each normal n_k and each point p:
    #   (cx + D*cos(n_k) - px)*cos(n_k) + (cy + D*sin(n_k) - py)*sin(n_k) >= 0
    #   (cx - D*cos(n_k) - px)*cos(n_k) + (cy - D*sin(n_k) - py)*sin(n_k) >= 0
    # Rearranging: -cx*cos(n_k) - cy*sin(n_k) - D <= -(px*cos(n_k) + py*sin(n_k))
    #              cx*cos(n_k) + cy*sin(n_k) - D <= px*cos(n_k) + py*sin(n_k)
    
    # Variables: [cx, cy, D]
    c_obj = np.array([0.0, 0.0, 1.0])
    
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in points:
            # Constraint 1: -nx*cx - ny*cy - D <= -(px*nx + py*ny)
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            # Constraint 2: nx*cx + ny*cy - D <= px*nx + py*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            # S = (D + apothem) / apothem, where apothem = sqrt(3)/2
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy), D
        else:
            return float('inf'), None, None
    except Exception:
        return float('inf'), None, None

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta_val=100.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    max_projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for nx, ny in normals:
            max_proj_this = -1e10
            for v in verts:
                dx = v[0] - ox
                dy = v[1] - oy
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                proj = lx * nx + ly * ny
                if proj > max_proj_this:
                    max_proj_this = proj
            max_projections.append(max_proj_this)
    
    if not max_projections:
        return 0.0
    
    max_proj = max(max_projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = [beta_val * (p - max_proj) for p in max_projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta_val
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
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
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts_list.append(get_hex_vertices(ix, iy, 1.0, math.radians(iangle)))
    
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
    """Check if the configuration is valid."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def objective(params, use_smooth=True, beta_val=100.0):
    """Objective function: outer side length + penalty for overlaps."""
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
    
    if use_smooth:
        S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta_val)
    else:
        S = compute_required_side(inner, (ox, oy), outer_angle)
    
    # Distance-based penalty for overlaps
    penalty = 0.0
    min_dist = SQRT3
    
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, _ = inner[i]
            cx2, cy2, _ = inner[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < min_dist:
                penalty += (min_dist - d) ** 2 * 50
    
    return S + penalty

def lp_objective(params):
    """LP-based objective: exact minimum enclosing hexagon side length."""
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
    
    S, _, _ = compute_required_side_lp(inner, (ox, oy), outer_angle)
    
    # Distance-based penalty for overlaps
    penalty = 0.0
    min_dist = SQRT3
    
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, _ = inner[i]
            cx2, cy2, _ = inner[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < min_dist:
                penalty += (min_dist - d) ** 2 * 50
    
    if S == float('inf'):
        return 1e6 + penalty
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Axial coordinates for 11 hexagons - multiple candidate configurations
    candidate_sets = [
        # Standard hexagonal cluster: center + ring 1 + partial ring 2
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, -1), (1, 1), (-2, 1), (-1, -1)],
        # Alternative ring 2 placement
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (0, 2), (0, -2)],
        # More linear arrangement
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (-1, 1), (3, 1)],
        # Compact 3x3+2
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (1, 3), (2, 3)],
        # Diamond-like
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3), (3, 2)]
    ]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    # Generate initial conditions from all candidate sets
    initial_conditions = []
    
    for axial_coords in candidate_sets:
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        # Standard orientation
        scale = 1.0 + 1e-6
        centers = [axial_to_cartesian(q, r, scale) for q, r in axial_coords]
        ic = np.zeros(3 * N + 3)
        for i, (px, py) in enumerate(centers):
            ic[3*i] = px
            ic[3*i+1] = py
            ic[3*i+2] = 0.0
        ic[3*N] = 0.0
        ic[3*N+1] = 0.0
        ic[3*N+2] = 0.0
        initial_conditions.append(ic)
        
        # Rotated outer by 30 degrees
        ic_rot = ic.copy()
        ic_rot[3*N+2] = np.radians(30)
        initial_conditions.append(ic_rot)
    
    # Add perturbed versions of the best candidate
    np.random.seed(42)
    for _ in range(3):
        base = initial_conditions[0].copy()
        perturbed = base + np.random.randn(3*N+3) * 0.03
        initial_conditions.append(perturbed)
    
    best_result = None
    best_score = float('inf')
    
    # Optimize each initial condition
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 25:
            break
        
        try:
            # Phase 1: LP-based optimization (exact objective, but non-smooth)
            # Use Nelder-Mead for the LP objective since it doesn't need gradients
            res_lp = minimize(lp_objective, ic,
                            method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            
            # Phase 2: Smooth refinement using L-BFGS-B
            if res_lp.fun < best_score or best_result is None:
                res_smooth = minimize(lambda p: objective(p, use_smooth=True, beta_val=50.0), res_lp.x,
                                    method='L-BFGS-B',
                                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                
                # Phase 3: Higher beta refinement
                res_smooth2 = minimize(lambda p: objective(p, use_smooth=True, beta_val=200.0), res_smooth.x,
                                      method='L-BFGS-B',
                                      options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10})
                
                # Phase 4: Exact objective refinement
                res_exact = minimize(lambda p: objective(p, use_smooth=False), res_smooth2.x,
                                    method='L-BFGS-B',
                                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10})
                
                # Pick the best
                candidates = [res_lp, res_smooth, res_smooth2, res_exact]
                best_res = min(candidates, key=lambda r: r.fun)
                
                if best_res.fun < best_score or best_result is None:
                    params = best_res.x
                    inner = []
                    for i in range(N):
                        cx = params[3*i]
                        cy = params[3*i+1]
                        ang = np.degrees(params[3*i+2]) % 360
                        inner.append((cx, cy, ang))
                    ox = params[3*N]
                    oy = params[3*N+1]
                    outer_angle = np.degrees(params[3*N+2]) % 360
                    
                    S, _, _ = compute_required_side_lp(inner, (ox, oy), outer_angle)
                    if S == float('inf'):
                        S = compute_required_side(inner, (ox, oy), outer_angle)
                    
                    if check_validity(inner, (ox, oy), S, outer_angle):
                        if S < best_score:
                            best_score = S
                            best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        axial_coords = candidate_sets[0][:N]
        centers_fb = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
        inner = [(px, py, 0.0) for px, py in centers_fb]
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Final polish with Nelder-Mead on LP objective
    if time.time() < deadline - 15 and best_result is not None:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res_nm = minimize(lp_objective, params_best,
                            method='Nelder-Mead',
                            options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_nm.fun < best_score:
                params = res_nm.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S, _, _ = compute_required_side_lp(inner, (ox, oy), outer_angle)
                if S == float('inf'):
                    S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
