"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
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

def compute_required_side_smooth(inner, outer_center, outer_angle_rad, beta=50.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    Takes outer_angle in radians for consistency.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Precompute normals in local frame
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(outer_angle_rad)
    sin_a = math.sin(outer_angle_rad)
    
    # Collect all projections
    n_projs = 6 * 6 * len(inner)
    projs = np.empty(n_projs)
    idx = 0
    for (ix, iy, iangle_rad) in inner:
        # Compute 6 vertices of unit hexagon
        cos_i = math.cos(iangle_rad)
        sin_i = math.sin(iangle_rad)
        # Vertex offsets from center: (cos(a), sin(a)) for a = iangle + k*pi/3
        for k in range(6):
            a = iangle_rad + k * math.pi / 3.0
            vx = ix + cos_i * math.cos(k * math.pi/3) - sin_i * math.sin(k * math.pi/3)
            vy = iy + sin_i * math.cos(k * math.pi/3) + cos_i * math.sin(k * math.pi/3)
            # Actually simpler: vertex = (ix + cos(a), iy + sin(a))
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                projs[idx] = lx * nx + ly * ny
                idx += 1
    
    max_proj = np.max(projs)
    if max_proj < -1000:
        return 0.0
    
    # Log-sum-exp for numerical stability
    scaled = beta * (projs - max_proj)
    sum_exp = np.sum(np.exp(scaled))
    smooth_max = max_proj + np.log(sum_exp) / beta
    
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

def objective(params, use_smooth=True, beta=50.0):
    """Objective function: outer side length + penalty for overlaps."""
    n = N
    inner_rad = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang_rad = params[3*i+2]
        inner_rad.append((cx, cy, ang_rad))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle_rad = params[3*n+2]
    
    if use_smooth:
        S = compute_required_side_smooth(inner_rad, (ox, oy), outer_angle_rad, beta)
    else:
        # Convert to degrees for compute_required_side
        inner_deg = [(cx, cy, np.degrees(ang_rad) % 360) for (cx, cy, ang_rad) in inner_rad]
        outer_angle_deg = np.degrees(outer_angle_rad) % 360
        S = compute_required_side(inner_deg, (ox, oy), outer_angle_deg)
    
    # Smooth penalty for overlaps using distance-based approximation
    penalty = 0.0
    for i in range(n):
        cx1, cy1, _ = inner_rad[i]
        for j in range(i+1, n):
            cx2, cy2, _ = inner_rad[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            # Minimum center distance for non-overlapping unit hexagons is 2*sqrt(3)/sqrt(3) = 2 (for aligned)
            # Actually for regular hexagons, min distance between centers = 2 * inradius = 2 * sqrt(3)/2 * side = sqrt(3) for touching
            # Wait: inradius of unit hexagon = sqrt(3)/2, so center-to-center for touching = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732
            # But if rotated, can be different. Use conservative estimate of 2.0
            min_d = 2.0
            if d < min_d:
                penalty += (min_d - d) ** 2 * 50.0
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    
    # Generate diverse initial conditions
    initial_conditions = []
    
    # Arrangement 1: Standard hexagonal lattice, scale 1.0
    scale1 = 1.0 + 1e-6
    centers1 = [axial_to_cartesian(q, r, scale1) for q, r in axial_coords]
    ic1 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(centers1):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    ic1[3*N] = 0.0
    ic1[3*N+1] = 0.0
    ic1[3*N+2] = 0.0
    initial_conditions.append(ic1)
    
    # Arrangement 2: Rotated outer hexagon by 30 degrees
    ic2 = ic1.copy()
    ic2[3*N+2] = np.radians(30)
    initial_conditions.append(ic2)
    
    # Arrangement 3: Slightly tighter scale
    scale3 = 0.98
    centers3 = [axial_to_cartesian(q, r, scale3) for q, r in axial_coords]
    ic3 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(centers3):
        ic3[3*i] = px
        ic3[3*i+1] = py
        ic3[3*i+2] = 0.0
    ic3[3*N] = 0.0
    ic3[3*N+1] = 0.0
    ic3[3*N+2] = 0.0
    initial_conditions.append(ic3)
    
    # Arrangement 4: Perturbed standard lattice
    np.random.seed(42)
    ic4 = ic1 + np.random.randn(3*N+3) * 0.05
    initial_conditions.append(ic4)
    
    # Arrangement 5: Different outer angle
    ic5 = ic1.copy()
    ic5[3*N+2] = np.radians(15)
    initial_conditions.append(ic5)
    
    # Arrangement 6: Rotated inner hexagons by 30 degrees
    ic6 = ic1.copy()
    for i in range(N):
        ic6[3*i+2] = np.radians(30)
    initial_conditions.append(ic6)
    
    # Arrangement 7: Different cluster shape - more compact
    # Try a different set of axial coordinates
    axial_coords2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (2, 0), (-1, -1), (-2, 0)
    ]
    if len(axial_coords2) > N:
        axial_coords2 = axial_coords2[:N]
    centers7 = [axial_to_cartesian(q, r, 1.0 + 1e-6) for q, r in axial_coords2]
    ic7 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(centers7):
        ic7[3*i] = px
        ic7[3*i+1] = py
        ic7[3*i+2] = 0.0
    ic7[3*N] = 0.0
    ic7[3*N+1] = 0.0
    ic7[3*N+2] = 0.0
    initial_conditions.append(ic7)
    
    # Arrangement 8: Random perturbation with different seed
    np.random.seed(123)
    ic8 = ic1 + np.random.randn(3*N+3) * 0.1
    initial_conditions.append(ic8)
    
    best_result = None
    best_score = float('inf')
    
    # Optimize each initial condition
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 20:
            break
        
        try:
            # First pass: smooth objective with moderate beta
            res = minimize(lambda p: objective(p, use_smooth=True, beta=50.0), ic, 
                          method='L-BFGS-B', 
                          options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Second pass: refine with exact objective
            if res.fun < best_score or best_result is None:
                res2 = minimize(lambda p: objective(p, use_smooth=False), res.x, 
                               method='L-BFGS-B', 
                               options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res2.fun < best_score:
                    params = res2.x
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
    
    # Refine if time permits
    if time.time() < deadline - 30:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            # Final refinement with exact objective
            res = minimize(lambda p: objective(p, use_smooth=False), params_best, 
                          method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            
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
