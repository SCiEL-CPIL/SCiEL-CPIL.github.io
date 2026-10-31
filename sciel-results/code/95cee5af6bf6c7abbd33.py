import numpy as np
import math
import time
from scipy.optimize import minimize, differential_evolution, linprog, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE = 145.0  # Safety margin before 150s kill


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
    inner: list of [x, y, angle_deg]
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


def sat_overlap_check(v1, v2, eps=1e-7):
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
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
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
    verts_cache = []
    for i in range(n):
        x, y, ang = inner[i]
        verts = get_hex_vertices(x, y, 1.0, math.radians(ang))
        verts_cache.append(verts)
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_cache[i], verts_cache[j], eps):
                return False
    return True


def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """
    Compute a smooth penalty for overlaps based on center distances.
    This is a fast approximation; full SAT is used for final validation.
    """
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
    return penalty


def construct():
    start_time = time.time()

    # 1. Define 11 hexagons on a hexagonal lattice (center + 10 neighbors)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]

    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    # 2. Convert axial to Cartesian
    # Flat-top hexagons: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    epsilon = 1e-6
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]

    # Verify initial layout validity
    if not check_all_disjoint(initial_inner):
        # Fallback to sparse linear layout if lattice fails (shouldn't happen with scale > 1)
        initial_inner = [[i * 2.5, 0.0, 0.0] for i in range(N)]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Calculate initial centroid for outer center guess
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # --- Phase 1: Optimize Outer Hexagon (Center and Angle) ---
    def objective_outer(params, inner_list):
        cx, cy, angle_deg = params
        side = compute_required_side(inner_list, (cx, cy), angle_deg)
        return side

    init_angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    best_outer_val = float('inf')
    best_outer_params = [cx_avg, cy_avg, 0.0]

    for init_angle in init_angles:
        if time.time() - start_time > DEADLINE - 60:
            break

        init_params = [cx_avg, cy_avg, float(init_angle)]
        try:
            res = minimize(
                lambda p: objective_outer(p, initial_inner),
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500}
            )
            if res.fun < best_outer_val:
                best_outer_val = res.fun
                best_outer_params = res.x
        except Exception:
            pass

    if best_outer_val < float('inf'):
        best_outer_center = (best_outer_params[0], best_outer_params[1])
        best_outer_angle = best_outer_params[2]
        best_side = best_outer_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # --- Phase 2: Joint Optimization using Differential Evolution ---
    # Variables: [x1, y1, a1, ..., xN, yN, aN, cx, cy, angle_outer]
    
    def objective_joint(params):
        if time.time() - start_time > DEADLINE - 10:
            return 1e10

        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append([x, y, a])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        side = compute_required_side(inner_list, (cx, cy), angle)

        # Penalty for overlaps (smooth approximation)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty

    # Define bounds for differential evolution
    # Inner hexagons: positions within a reasonable range, angles 0-360
    # Outer center: reasonable range, angle 0-360
    bounds = []
    for i in range(N):
        bounds.extend([(-5, 5), (-5, 5), (0, 360)])
    bounds.extend([(-5, 5), (-5, 5), (0, 360)])

    # Run differential evolution if time permits
    if time.time() - start_time < DEADLINE - 30:
        try:
            # Use a smaller population to save time
            de_result = differential_evolution(
                objective_joint,
                bounds,
                seed=42,
                maxiter=10,
                popsize=10,
                tol=1e-4,
                polish=False
            )
            
            if de_result.fun < best_side:
                # Extract new best
                new_inner = []
                for i in range(N):
                    new_inner.append([de_result.x[i * 3], de_result.x[i * 3 + 1], de_result.x[i * 3 + 2]])
                
                # Verify validity
                if check_all_disjoint(new_inner):
                    new_side = compute_required_side(new_inner, (de_result.x[N * 3], de_result.x[N * 3 + 1]), de_result.x[N * 3 + 2])
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = new_inner
                        best_outer_center = (de_result.x[N * 3], de_result.x[N * 3 + 1])
                        best_outer_angle = de_result.x[N * 3 + 2]
        except Exception:
            pass

    # --- Phase 3: Local Polish with Nelder-Mead ---
    # Initialize params from best found so far
    params = []
    for i in range(N):
        params.extend(best_inner[i])
    params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

    # Run a few optimization steps
    num_rounds = 3
    for r in range(num_rounds):
        if time.time() - start_time > DEADLINE - 10:
            break

        try:
            res = minimize(
                objective_joint,
                params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 2000}
            )
            if res.fun < best_side:
                # Extract new best
                new_inner = []
                for i in range(N):
                    new_inner.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
                
                # Verify validity
                if check_all_disjoint(new_inner):
                    new_side = compute_required_side(new_inner, (res.x[N * 3], res.x[N * 3 + 1]), res.x[N * 3 + 2])
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = new_inner
                        best_outer_center = (res.x[N * 3], res.x[N * 3 + 1])
                        best_outer_angle = res.x[N * 3 + 2]
                        params = res.x
        except Exception:
            pass

    # --- Phase 4: LP-based refinement (from Inspiration 1) ---
    # If time permits, use linear programming to refine the outer hexagon
    # given the current inner positions. This exploits the fact that for fixed
    # inner hexagons, the outer containment is a linear problem in (cx, cy, D).
    
    if time.time() - start_time < DEADLINE - 15:
        try:
            # Extract current inner positions
            pts = np.array([[p[0], p[1]] for p in best_inner])
            inner_angles = [p[2] for p in best_inner]
            
            # For each possible outer angle, solve LP to find optimal center and D
            # D = S * APOTHEM - APOTHEM (minimum distance from outer center to inner hexagon boundary)
            
            def calc_S_for_beta_lp(beta):
                # Check time limit
                if time.time() - start_time > DEADLINE - 10:
                    return float('inf'), None
                
                normals = []
                for k in range(6):
                    ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                    normals.append((math.cos(ang), math.sin(ang)))
                
                # Variables: cx, cy, D
                # Minimize D
                c_obj = np.array([0.0, 0.0, 1.0])
                
                A_ub = []
                b_ub = []
                
                # For each inner hexagon, we need to consider its vertices
                # projected onto the outer normals
                for i in range(N):
                    ix, iy, iang = best_inner[i]
                    verts = get_hex_vertices(ix, iy, 1.0, math.radians(iang))
                    
                    for nx, ny in normals:
                        for vx, vy in verts:
                            # Constraint: (vx - cx)nx + (vy - cy)ny <= D
                            # -nx*cx - ny*cy - D <= -(vx*nx + vy*ny)
                            A_ub.append([-nx, -ny, -1.0])
                            b_ub.append(-(vx * nx + vy * ny))
                            
                            # Constraint: -((vx - cx)nx + (vy - cy)ny) <= D
                            # nx*cx + ny*cy - D <= vx*nx + vy*ny
                            A_ub.append([nx, ny, -1.0])
                            b_ub.append(vx * nx + vy * ny)
                
                A_ub = np.array(A_ub)
                b_ub = np.array(b_ub)
                
                # Bounds
                # D must be >= 0
                bounds_lp = [(None, None), (None, None), (0, None)]
                
                try:
                    res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
                    if res_lp.status == 0:
                        D = res_lp.x[2]
                        cx = res_lp.x[0]
                        cy = res_lp.x[1]
                        S = (D + APOTHEM) / APOTHEM
                        return S, (cx, cy)
                    else:
                        return float('inf'), None
                except Exception:
                    return float('inf'), None
            
            # Sample beta values
            best_S_lp = float('inf')
            best_beta_lp = 0.0
            best_center_lp = None
            
            for i in range(25):
                beta = i * (math.pi / 6) / 24
                S, center = calc_S_for_beta_lp(beta)
                if S < best_S_lp:
                    best_S_lp = S
                    best_beta_lp = beta
                    best_center_lp = center
            
            # Refine with minimize_scalar
            if time.time() - start_time < DEADLINE - 10:
                try:
                    res_scalar = minimize_scalar(
                        lambda b: calc_S_for_beta_lp(b)[0],
                        bounds=(0, math.pi / 6),
                        method='bounded',
                        options={'maxiter': 50, 'xatol': 1e-8}
                    )
                    if res_scalar.fun < best_S_lp:
                        best_S_lp = res_scalar.fun
                        best_beta_lp = res_scalar.x
                        _, best_center_lp = calc_S_for_beta_lp(best_beta_lp)
                except Exception:
                    pass
            
            if best_center_lp is not None and best_S_lp < best_side:
                best_side = best_S_lp
                best_outer_center = best_center_lp
                best_outer_angle = math.degrees(best_beta_lp)
        except Exception:
            pass

    # Final verification and cleanup
    # Ensure we return a valid configuration
    if not check_all_disjoint(best_inner):
        # Fallback to initial if optimization broke it
        best_inner = [list(x) for x in initial_inner]
        best_side = compute_required_side(best_inner, (cx_avg, cy_avg), 0.0)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0

    return best_inner, best_outer_center, best_side, best_outer_angle
