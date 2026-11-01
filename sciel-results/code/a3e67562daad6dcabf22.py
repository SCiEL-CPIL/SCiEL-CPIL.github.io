"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# Precompute hexagon vertex offsets for unit hexagon at angle 0
_HEX_ANGLES = np.pi / 3.0 * np.arange(6)
_HEX_COS = np.cos(_HEX_ANGLES)
_HEX_SIN = np.sin(_HEX_ANGLES)
# Outer hexagon support normals (inward-pointing)
_OUTER_NORMALS = np.array([[math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] for i in range(6)])

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + _HEX_ANGLES
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        ang = math.radians(iangle)
        ca, sa = math.cos(ang), math.sin(ang)
        # Vertices of unit hexagon
        for k in range(6):
            vx = _HEX_COS[k]
            vy = _HEX_SIN[k]
            # Rotate vertex
            rvx = vx * ca - vy * sa
            rvy = vx * sa + vy * ca
            # Translate
            wx = ix + rvx
            wy = iy + rvy
            # Transform to outer frame
            dx = wx - ox
            dy = wy - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            # Project onto all 6 normals
            for m in range(6):
                proj = lx * _OUTER_NORMALS[m, 0] + ly * _OUTER_NORMALS[m, 1]
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    This provides better gradient information for optimization.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    n_inner = len(inner)
    projections = np.empty(n_inner * 6 * 6)
    idx = 0
    for (ix, iy, iangle) in inner:
        ang = math.radians(iangle)
        ca, sa = math.cos(ang), math.sin(ang)
        for k in range(6):
            vx = _HEX_COS[k]
            vy = _HEX_SIN[k]
            rvx = vx * ca - vy * sa
            rvy = vx * sa + vy * ca
            wx = ix + rvx
            wy = iy + rvy
            dx = wx - ox
            dy = wy - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for m in range(6):
                projections[idx] = lx * _OUTER_NORMALS[m, 0] + ly * _OUTER_NORMALS[m, 1]
                idx += 1
    
    max_proj = np.max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = beta * (projections - max_proj)
    # Clip to avoid overflow in exp
    scaled = np.clip(scaled, -500, 500)
    # Use log-sum-exp trick
    sum_exp = np.sum(np.exp(scaled))
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
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
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def objective(params, use_smooth=True, beta=100.0):
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
        S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta)
    else:
        S = compute_required_side(inner, (ox, oy), outer_angle)
    
    penalty = 0.0
    # Quick distance-based pre-check before expensive SAT
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, _ = inner[i]
            cx2, cy2, _ = inner[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.05:
                # Only do full SAT if close enough
                v1 = get_hex_vertices(cx1, cy1, 1.0, math.radians(inner[i][2]))
                v2 = get_hex_vertices(cx2, cy2, 1.0, math.radians(inner[j][2]))
                if sat_overlap_check(v1, v2, eps=1e-8):
                    # Penalty grows quadratically with overlap depth
                    # Use center distance as proxy for overlap severity
                    overlap_depth = max(0.0, 2.0 - d)
                    penalty += overlap_depth ** 2 * 50.0
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Axial coordinates for 11 hexagons in a compact cluster
    # Center + 6 in ring 1 + 4 in ring 2
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    def make_params(centers, outer_center=(0.0, 0.0), outer_angle=0.0, inner_angles=None):
        """Helper to build parameter vector."""
        params = np.zeros(3 * N + 3)
        for i, (px, py) in enumerate(centers):
            params[3*i] = px
            params[3*i+1] = py
            if inner_angles is not None:
                params[3*i+2] = np.radians(inner_angles[i])
        params[3*N] = outer_center[0]
        params[3*N+1] = outer_center[1]
        params[3*N+2] = np.radians(outer_angle)
        return params
    
    # Generate diverse initial conditions
    initial_conditions = []
    
    # Base lattice at various scales
    for scale in [1.0 + 1e-6, 0.99, 0.98, 0.97]:
        centers = [axial_to_cartesian(q, r, scale) for q, r in axial_coords]
        ic = make_params(centers)
        initial_conditions.append(ic)
    
    # Different outer rotations
    for oa in [0, 15, 30, 45, 60]:
        centers = [axial_to_cartesian(q, r, 1.0 + 1e-6) for q, r in axial_coords]
        ic = make_params(centers, outer_angle=oa)
        initial_conditions.append(ic)
    
    # Rotated inner hexagons
    for ia in [10, 15, 30]:
        centers = [axial_to_cartesian(q, r, 1.0 + 1e-6) for q, r in axial_coords]
        inner_angles = [ia] * N
        ic = make_params(centers, inner_angles=inner_angles)
        initial_conditions.append(ic)
    
    # Mixed rotations
    centers = [axial_to_cartesian(q, r, 1.0 + 1e-6) for q, r in axial_coords]
    inner_angles = [0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0]
    ic = make_params(centers, inner_angles=inner_angles)
    initial_conditions.append(ic)
    
    # Random perturbations with multiple seeds
    np.random.seed(42)
    base_centers = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        perturbed = [(x + np.random.randn() * 0.05, y + np.random.randn() * 0.05) for x, y in base_centers]
        ic = make_params(perturbed)
        initial_conditions.append(ic)
    
    # Shifted center configurations
    for shift in [(0.1, 0.1), (-0.1, 0.1), (0.1, -0.1), (0.0, 0.2), (0.2, 0.0)]:
        centers = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
        ic = make_params(centers, outer_center=shift)
        initial_conditions.append(ic)
    
    best_result = None
    best_score = float('inf')
    all_candidates = []  # Store all promising candidates for later refinement
    
    # Optimize each initial condition with adaptive beta
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 20:
            break
        
        try:
            # Adaptive beta: start lower, increase for refinement
            beta_seq = [30.0, 50.0, 80.0]
            
            for beta in beta_seq:
                if time.time() > deadline - 20:
                    break
                
                res = minimize(lambda p: objective(p, use_smooth=True, beta=beta), ic, 
                              method='L-BFGS-B', 
                              options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8})
                
                # Extract result
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    all_candidates.append((S, inner, (ox, oy), outer_angle))
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        centers_fb = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
        inner = [(px, py, 0.0) for px, py in centers_fb]
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Refine top candidates with exact objective
    all_candidates.sort(key=lambda x: x[0])
    top_candidates = all_candidates[:min(5, len(all_candidates))]
    
    for S_cand, inner_cand, center_cand, angle_cand in top_candidates:
        if time.time() > deadline - 15:
            break
        
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(inner_cand):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = center_cand[0]
            params_best[3*N+1] = center_cand[1]
            params_best[3*N+2] = np.radians(angle_cand)
            
            # High-precision refinement with exact objective
            res = minimize(lambda p: objective(p, use_smooth=False), params_best, 
                          method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Final ultra-precise polish on best result
    if time.time() < deadline - 8 and best_result is not None:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res = minimize(lambda p: objective(p, use_smooth=False), params_best, 
                          method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
