"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Smooth approximation using log-sum-exp over ALL vertex projections.
    This properly approximates the global max projection.
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
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
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
    verts_list = [get_hex_vertices(ix, iy, 1.0, math.radians(ia)) for (ix, iy, ia) in inner]
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
    """Check validity: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def objective(params, use_smooth=True, beta=100.0):
    """Objective: outer side length + penalty for overlaps."""
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
    min_dist = SQRT3
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, _ = inner[i]
            cx2, cy2, _ = inner[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < min_dist:
                penalty += (min_dist - d) ** 2 * 50
    return S + penalty

def axial_to_cartesian(q, r, scale=1.0):
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return (x, y)

def make_initial_params(centers, outer_center=(0.0, 0.0), outer_angle=0.0, inner_angles=None):
    """Create parameter vector from centers and angles."""
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

def params_to_result(params):
    """Convert params to (inner, center, side, angle) result."""
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
    return (inner, (ox, oy), S, outer_angle)

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Define multiple distinct lattice topologies for N=11
    # Each is a set of axial coordinates (q, r)
    
    # Topology 1: Flower (center + ring1 + 4 from ring2)
    topo_flower = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    # Topology 2: 3-5-3 row arrangement
    topo_353 = [
        (0, 0), (1, 0), (2, 0),       # row 0: 3 hexes
        (-1, 1), (0, 1), (1, 1), (2, 1), (3, 1),  # row 1: 5 hexes
        (0, 2), (1, 2), (2, 2)         # row 2: 3 hexes
    ]
    
    # Topology 3: 4-3-4 row arrangement
    topo_434 = [
        (0, 0), (1, 0), (2, 0), (3, 0),       # row 0: 4 hexes
        (-1, 1), (0, 1), (1, 1),              # row 1: 3 hexes
        (0, 2), (1, 2), (2, 2), (3, 2)        # row 2: 4 hexes
    ]
    
    # Topology 4: 5-3-3 compact
    topo_533 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),   # row 0: 5 hexes
        (-1, 1), (0, 1), (1, 1),                   # row 1: 3 hexes
        (0, 2), (1, 2), (2, 2)                     # row 2: 3 hexes
    ]
    
    # Topology 5: Diagonal compact
    topo_diag = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3)
    ]
    
    topologies = [topo_flower, topo_353, topo_434, topo_533, topo_diag]
    
    # Generate initial conditions from each topology with variations
    initial_conditions = []
    
    for topo in topologies:
        if len(topo) > N:
            topo = topo[:N]
        
        # Standard scale
        scale = 1.0 + 1e-6
        centers = [axial_to_cartesian(q, r, scale) for q, r in topo]
        
        # Variation 1: Standard
        ic = make_initial_params(centers)
        initial_conditions.append(ic)
        
        # Variation 2: Rotated outer by 30 deg
        ic2 = make_initial_params(centers, outer_angle=30.0)
        initial_conditions.append(ic2)
        
        # Variation 3: Slightly compressed
        scale_comp = 0.99
        centers_comp = [axial_to_cartesian(q, r, scale_comp) for q, r in topo]
        ic3 = make_initial_params(centers_comp)
        initial_conditions.append(ic3)
        
        # Variation 4: Rotated outer by 15 deg
        ic4 = make_initial_params(centers, outer_angle=15.0)
        initial_conditions.append(ic4)
        
        # Variation 5: With perturbation
        np.random.seed(hash(str(topo)) % 10000)
        ic5 = ic + np.random.randn(3*N+3) * 0.03
        initial_conditions.append(ic5)
    
    best_result = None
    best_score = float('inf')
    
    def try_optimize(ic, label=""):
        nonlocal best_result, best_score
        if time.time() > deadline - 25:
            return
        
        try:
            # Multi-stage optimization
            # Stage 1: Smooth with moderate beta
            res1 = minimize(lambda p: objective(p, use_smooth=True, beta=30.0), ic,
                           method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if time.time() > deadline - 20:
                return
            
            # Stage 2: Smooth with higher beta
            res2 = minimize(lambda p: objective(p, use_smooth=True, beta=100.0), res1.x,
                           method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if time.time() > deadline - 15:
                return
            
            # Stage 3: Exact objective
            res3 = minimize(lambda p: objective(p, use_smooth=False), res2.x,
                           method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            # Use the best of the three
            candidates = [res1, res2, res3]
            best_res = min(candidates, key=lambda r: r.fun)
            
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
            S = compute_required_side(inner, (ox, oy), outer_angle)
            
            if check_validity(inner, (ox, oy), S, outer_angle):
                if S < best_score:
                    best_score = S
                    best_result = (inner, (ox, oy), S, outer_angle)
                    print(f"  {label}: S={S:.6f}")
        except Exception as e:
            pass
    
    # Optimize each initial condition
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 30:
            break
        try_optimize(ic, f"init{idx}")
    
    # Fallback if no valid result
    if best_result is None:
        centers_fb = [axial_to_cartesian(q, r, 1.0) for q, r in topo_flower]
        inner = [(px, py, 0.0) for px, py in centers_fb]
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.01
        best_result = (inner, (0.0, 0.0), S, 0.0)
        best_score = S
    
    # Differential Evolution global search
    if time.time() < deadline - 50:
        try:
            bounds = [(-5, 5)] * (3 * N) + [(-5, 5), (-5, 5), (0, 2 * math.pi)]
            
            def de_objective(params):
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                
                # Quick rejection: if centers are too far apart
                max_d = 0
                for i in range(N):
                    for j in range(i+1, N):
                        d = math.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
                        if d > 8:
                            return 100.0
                
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if not check_all_disjoint(inner):
                    return S + 50.0
                
                if not check_inside(inner, (ox, oy), S, outer_angle):
                    return S + 50.0
                
                return S
            
            res_de = differential_evolution(de_objective, bounds,
                                           maxiter=30, popsize=8,
                                           tol=1e-8, seed=42,
                                           polish=False)
            
            if res_de.fun < best_score:
                params = res_de.x
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
                    print(f"  DE: S={S:.6f}")
        except Exception:
            pass
    
    # Final refinement with Nelder-Mead on best result
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
            
            # L-BFGS-B refinement
            res_l = minimize(lambda p: objective(p, use_smooth=False), params_best,
                            method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res_l.fun < best_score:
                params = res_l.x
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
                    print(f"  L-BFGS refine: S={S:.6f}")
        except Exception:
            pass
    
    # Nelder-Mead polish
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
            
            res_nm = minimize(lambda p: objective(p, use_smooth=False), params_best,
                            method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-13})
            
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
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
                    best_score = S
                    print(f"  NM polish: S={S:.6f}")
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
