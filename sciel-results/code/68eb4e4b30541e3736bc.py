import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 140.0  # Safety margin before 150s kill

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

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
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    
    # 1. Define 11 hexagons on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    # 2. Convert axial to Cartesian
    epsilon = 1e-6
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Calculate initial centroid for outer center guess
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # --- Phase 1: Optimize Outer Hexagon (Center and Angle) ---
    # Variables: [cx, cy, angle_deg]
    
    def objective_outer(params, inner_list):
        cx, cy, angle_deg = params
        side = compute_required_side(inner_list, (cx, cy), angle_deg)
        return side

    init_angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    best_outer_val = float('inf')
    best_outer_params = [cx_avg, cy_avg, 0.0]

    for init_angle in init_angles:
        if time.time() - start_time > DEADLINE - 40:
            break
        
        init_params = [cx_avg, cy_avg, float(init_angle)]
        try:
            res = minimize(
                lambda p: objective_outer(p, initial_inner),
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 500}
            )
            if res.fun < best_outer_val:
                best_outer_val = res.fun
                best_outer_params = res.x
        except Exception:
            pass

    if best_outer_params is not None:
        best_outer_center = (best_outer_params[0], best_outer_params[1])
        best_outer_angle = best_outer_params[2]
        best_side = best_outer_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # --- Phase 2: Joint Optimization ---
    # We will optimize inner positions and the outer center/angle jointly.
    # To handle the non-convexity and potential overlaps, we use a penalty-based objective.
    # Variables: 
    #   Inner: x_i, y_i, a_i (discretized angle index handled outside or via penalty?)
    #   Actually, optimizing continuous angles in NM is hard. 
    #   Strategy: Keep angles fixed at 0 for the joint optimization of positions, 
    #   as the lattice packing is already quite tight. 
    #   Alternatively, try a few fixed angle configurations for the whole set.
    
    # Let's try optimizing positions (x,y) for all N hexagons and outer (cx, cy, angle).
    # Total vars: 2*N + 3 = 25.
    
    def objective_joint(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e10
            
        # Extract inner positions
        inner_list = []
        for i in range(N):
            x = params[i * 2]
            y = params[i * 2 + 1]
            # Keep angle at 0 for stability in this phase, or allow small rotation?
            # Let's stick to 0 for now to ensure disjointness is easier to maintain via distance penalty
            inner_list.append([x, y, 0.0])

        # Extract outer params
        cx = params[2 * N]
        cy = params[2 * N + 1]
        angle = params[2 * N + 2]

        side = compute_required_side(inner_list, (cx, cy), angle)
        
        # Penalty for overlaps: 
        # For two unit hexagons to be disjoint, their centers must be at least 
        # sqrt(3) apart if aligned, or slightly less if rotated.
        # Using center distance > sqrt(3) - epsilon is a safe sufficient condition for aligned hexagons.
        penalty = 0.0
        min_dist = SQRT3 - 0.01 
        for i in range(N):
            xi, yi, _ = inner_list[i]
            for j in range(i + 1, N):
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < min_dist:
                    # Quadratic penalty
                    penalty += (min_dist - dist) ** 2 * 100.0

        return side + penalty

    # Initial parameters for joint optimization
    init_params_joint = []
    for x, y, _ in initial_inner:
        init_params_joint.append(x)
        init_params_joint.append(y)
    init_params_joint.append(best_outer_center[0])
    init_params_joint.append(best_outer_center[1])
    init_params_joint.append(best_outer_angle)

    # Run joint optimization
    try:
        if time.time() - start_time < DEADLINE - 10:
            res_joint = minimize(
                objective_joint,
                init_params_joint,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 2000}
            )
            
            if res_joint.fun < best_side + 1e-4: # Only accept if better or very close (penalty might be 0)
                # We need to verify if the penalty is actually 0 (no overlaps)
                # Reconstruct inner from result
                new_inner = []
                for i in range(N):
                    new_inner.append([res_joint.x[i*2], res_joint.x[i*2+1], 0.0])
                
                new_cx = res_joint.x[2*N]
                new_cy = res_joint.x[2*N+1]
                new_angle = res_joint.x[2*N+2]
                
                # Check for overlaps explicitly
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dist = math.sqrt((new_inner[i][0]-new_inner[j][0])**2 + (new_inner[i][1]-new_inner[j][1])**2)
                        if dist < SQRT3 - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    new_side = compute_required_side(new_inner, (new_cx, new_cy), new_angle)
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = new_inner
                        best_outer_center = (new_cx, new_cy)
                        best_outer_angle = new_angle
    except Exception:
        pass

    # Final safety check and cleanup
    if best_side == float('inf'):
        # Fallback to initial
        best_inner = initial_inner
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle
