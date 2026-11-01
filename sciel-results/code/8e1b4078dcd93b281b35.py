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
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    # Log-sum-exp for smooth max
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    # Smooth max via log-sum-exp: (1/beta) * log(sum(exp(beta * p_i)))
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
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
    Compute a smooth penalty for overlapping inner hexagons.
    Uses a soft measure based on center distance, which is a proxy for overlap
    for hexagons of similar orientation. For robustness, we use a smooth min-distance
    approximation if centers are close.
    """
    n = len(inner)
    penalty = 0.0
    for i in range(n):
        cx1, cy1, a1 = inner[i]
        for j in range(i + 1, n):
            cx2, cy2, a2 = inner[j]
            d = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # Minimum distance for non-overlapping hexagons is roughly 2*sqrt(3) ~ 3.464
            # But depends on angle. A safe lower bound for any orientation is 2.0 (diameter)
            # Actually, for hexagons, min distance between centers if touching is 2*sqrt(3) if aligned,
            # but can be less if rotated? No, the width is 2*sqrt(3) and height is 2.
            # The minimal distance between centers of two non-overlapping unit hexagons is 2.0 (if they touch at vertices? No, vertex to vertex is 2*side=2).
            # Wait, the distance from center to vertex is 1. So two hexagons touching vertex-to-vertex have center distance 2.
            # Two hexagons touching side-to-side have center distance 2*sqrt(3) ~ 3.464.
            # So 2.0 is the absolute minimum center distance for non-overlap.
            
            threshold = 2.0
            if d < threshold + 0.5:
                # Smooth penalty: exp(-beta * (threshold - d)) if d < threshold
                # Or quadratic: (threshold - d)^2
                # Let's use a smooth exponential decay for better gradients
                if d < threshold:
                    penalty += math.exp(-beta * (threshold - d))
                else:
                    # Small penalty to encourage separation even if not overlapping
                    penalty += 0.1 * (threshold - d)**2 if d < threshold + 0.1 else 0
    
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
    for _ in range(2):
        perturbed = params0 + np.random.randn(3*N+3) * 0.05
        # Keep angles small
        for i in range(N):
            perturbed[3*i+2] = np.random.randn() * 0.1
        initial_conditions.append(perturbed)
    
    for ic in initial_conditions:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(objective, ic, args=(deadline,), method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
            
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
