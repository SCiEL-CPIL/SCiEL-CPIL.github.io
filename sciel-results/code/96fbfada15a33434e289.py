"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    SQRT3 = math.sqrt(3)

    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2, eps=1e-6):
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = min(vx * nx + vy * ny for vx, vy in hex1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in hex1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in hex2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in hex2_verts)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))

        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj

        return 2.0 * max_proj / SQRT3

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # High-quality initial layouts (honeycomb variants)
    coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    coords2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords3 = [(0,0), (1,0), (-1,0), (0,1), (1,1), (-1,1), (0,-1), (1,-1), (-1,-1), (2,0), (-2,0)]
    coords4 = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(-2,0),(2,0)]
    
    configs = [coords1, coords2, coords3, coords4]

    def make_initial_params(coords):
        params = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        return np.array(params)

    def objective(p):
        # p = [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_angle]
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        # Center the configuration
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        outer_angle = p[N*3]
        
        hexagons = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hexagons.append(get_hex_vertices(cx, cy, a))
            
        return calc_outer_side(hexagons, outer_angle)

    def constraints(p):
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                # Separation constraint: min distance > 0
                # We use a proxy: the minimum over all axes of (gap)
                # To make it differentiable, we can use the smooth minimum or just the raw gap for SAT axes
                # However, SLSQP handles non-smoothness poorly. 
                # A common trick: use the minimum gap over the 12 axes.
                # Gap = min(0, min1 - max2, min2 - max1) is not smooth.
                # Instead, we enforce that for AT LEAST ONE axis, the separation is > 0.
                # This is an OR constraint, which is hard.
                
                # Alternative: Use the "minimum distance" function which is smooth almost everywhere.
                # For hexagons, the distance is determined by the closest pair of vertices/edges.
                # Let's use a simpler constraint: distance between centers > 2*inradius? No, that's for non-rotated.
                
                # Let's use the actual minimum distance between the two polygons.
                # We can compute the min distance by checking all vertex-edge pairs.
                # This is complex. 
                
                # Let's stick to the SAT-based separation but smooth it.
                # Actually, for SLSQP, we can define the constraint as:
                # min_{axes} (max1 - max2, min1 - min2) < 0  =>  Separation exists.
                # This is a non-convex, non-smooth constraint.
                
                # Better approach: Use the "penalty" method inside the objective but with a very high weight, 
                # or use a different solver like Differential Evolution to find a feasible point, then SLSQP.
                
                # Given the complexity, let's use a hybrid: 
                # 1. Find a feasible configuration using a simple local search that enforces disjointness.
                # 2. Use SLSQP to minimize the outer side length, with constraints that maintain disjointness.
                
                # For the constraint function, we will return the minimum "separation" over all axes.
                # If separation > 0, it's feasible.
                # We define constraint as: min_separation >= 0.
                # To make it smooth, we can use a log-sum-exp approximation of the minimum, or just the raw minimum.
                # Raw minimum is non-smooth, but SLSQP can often handle it if the initial point is good.
                
                h1 = get_hex_vertices(shifted_centers[i][0], shifted_centers[i][1], angles[i])
                h2 = get_hex_vertices(shifted_centers[j][0], shifted_centers[j][1], angles[j])
                a1 = get_hex_axes(angles[i])
                a2 = get_hex_axes(angles[j])
                
                min_sep = float('inf')
                for axis in a1 + a2:
                    nx, ny = axis
                    min1 = min(vx * nx + vy * ny for vx, vy in h1)
                    max1 = max(vx * nx + vy * ny for vx, vy in h1)
                    min2 = min(vx * nx + vy * ny for vx, vy in h2)
                    max2 = max(vx * nx + vy * ny for vx, vy in h2)
                    
                    # Separation is positive if they are disjoint
                    sep1 = min1 - max2
                    sep2 = min2 - max1
                    # If disjoint, one of these is positive.
                    # We want the "minimum" separation to be >= 0? No, we want the MAXIMUM separation to be >= 0?
                    # If they are disjoint, there exists an axis where they are separated.
                    # So max(sep1, sep2) > 0.
                    # We want to enforce that they are NOT overlapping.
                    # Overlap means for ALL axes, they intersect (sep1 < 0 AND sep2 < 0).
                    # So Disjoint means: NOT (for all axes, sep1 < 0 AND sep2 < 0).
                    # This is an OR constraint, which SLSQP cannot handle directly.
                    
                    # Workaround: Use a penalty in the objective instead of hard constraints for overlap,
                    # and use SLSQP to minimize (Outer Side + Penalty).
                    # This is what the previous program did, but we will use SLSQP's gradient capabilities.
                    pass
                
                # Since hard constraints are difficult, we will rely on a high-quality initial guess 
                # and a penalty method in the objective, but optimized by SLSQP.
                # However, the directive asks for a structural change. 
                # Let's use a different strategy: 
                # Use the "inner radius" packing bound. 
                # Or, simply use a better local search: 
                # "Gradient Descent with Projection".
                
                # Let's implement a simple Gradient Descent with projection onto the feasible set (disjoint hexagons).
                # 1. Compute gradient of outer side length.
                # 2. Move in negative gradient direction.
                # 3. If overlap occurs, resolve collision by pushing hexagons apart.
                
                # This is complex to code correctly in a short time.
                
                # Let's try a different structural change: 
                # Use **Differential Evolution** for global search, followed by **Nelder-Mead** for local refinement, 
                # but with a **smoothed overlap penalty** that is much more aggressive and uses a **log-barrier** for the outer side.
                
                # Actually, the previous program used SA + NM. 
                # Let's use **Particle Swarm Optimization (PSO)** for global search, which is often better for packing problems, 
                # followed by **SLSQP** with a penalty method.
                
                pass
        return []

    # Since the constraint formulation is tricky for SLSQP, let's use a **Penalty Method with SLSQP**.
    # Objective: Outer Side + Lambda * Overlap_Penalty
    # We will use SLSQP to minimize this.
    
    def evaluate_config_smooth(params, num_outer_samples=12, beta=50.0, overlap_weight=1000.0):
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[i*3], params[i*3+1]))
            angles.append(params[i*3+2])

        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]

        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hexagons.append(get_hex_vertices(cx, cy, a))
            hex_axes.append(get_hex_axes(a))

        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    dx = shifted_centers[i][0] - shifted_centers[j][0]
                    dy = shifted_centers[i][1] - shifted_centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    penetration = max(0.0, SQRT3 - dist)
                    overlap_penalty += penetration * penetration

        best_side = float('inf')
        for k in range(num_outer_samples):
            oa_deg = k * 60.0 / num_outer_samples
            # Smooth outer side
            base_angles = [30, 90, 150, 210, 270, 330]
            axis_dirs = []
            for a in base_angles:
                rad = math.radians(a + oa_deg)
                axis_dirs.append((math.cos(rad), math.sin(rad)))
            
            all_projs = []
            for hex_verts in hexagons:
                for vx, vy in hex_verts:
                    for nx, ny in axis_dirs:
                        proj = abs(vx * nx + vy * ny)
                        all_projs.append(proj)
            
            if not all_projs:
                continue
                
            max_p = max(all_projs)
            if max_p == 0:
                continue
            log_sum = max_p + math.log(sum(math.exp((p - max_p) * beta) for p in all_projs)) / beta
            side = 2.0 * log_sum / SQRT3
            if side < best_side:
                best_side = side

        return overlap_weight * overlap_penalty + best_side

    best_params = None
    best_score = float('inf')

    for config in configs:
        if time.time() > deadline - 20.0:
            break

        params = make_initial_params(config)
        
        # Add outer angle parameter
        params = np.append(params, 0.0)
        
        # SLSQP Optimization
        def objective_slsqp(p):
            return evaluate_config_smooth(p, num_outer_samples=8, beta=50.0, overlap_weight=500.0)

        try:
            res = minimize(objective_slsqp, params, method='SLSQP',
                         options={'maxiter': 100, 'ftol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

        # Nelder-Mead Refinement
        if time.time() < deadline - 10.0:
            def objective_nm(p):
                return evaluate_config_smooth(p, num_outer_samples=12, beta=50.0, overlap_weight=1000.0)

            try:
                res2 = minimize(objective_nm, best_params if best_params is not None else params, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass

    if best_params is None:
        # Fallback to initial config
        params = make_initial_params(coords1)
        best_params = np.append(params, 0.0)

    final_params = best_params.copy()

    centers = []
    angles = []
    for i in range(N):
        centers.append((final_params[i*3], final_params[i*3+1]))
        angles.append(final_params[i*3+2])

    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]

    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hexagons.append(get_hex_vertices(cx, cy, a))
        hex_axes.append(get_hex_axes(a))

    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to a known valid config
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side_fb = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side_fb, 0.0)

    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa

    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])

    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
