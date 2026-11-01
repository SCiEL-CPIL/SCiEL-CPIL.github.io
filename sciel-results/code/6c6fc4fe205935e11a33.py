import numpy as np
import math
import time
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 140.0  # Safety margin before 150s kill


def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])


def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    # Normals in outer frame (30 + i*60 degrees)
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        # Transform to outer local frame
        local = (verts - np.array([ox, oy])) @ R.T
        # Project onto all 6 normals
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    return 2.0 * max_proj / SQRT3


def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth (log-sum-exp) surrogate for the max projection.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    return 2.0 * smooth_max / SQRT3


def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True


def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True


def overlap_penalty(inner, min_dist=SQRT3 - 0.004):
    """Quadratic penalty when center distances fall below min_dist."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen


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
    if not check_disjoint(initial_inner):
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
        side = compute_required_side_fast(inner_list, (cx, cy), angle_deg)
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
        best_side = compute_required_side_fast(initial_inner, best_outer_center, best_outer_angle)

    # --- Phase 2: Joint Optimization with Smooth Surrogate ---
    # We use a smooth approximation for the objective to allow gradient-based methods
    # or more effective Nelder-Mead steps.
    
    def objective_joint_smooth(params, beta=50.0):
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

        side = compute_required_side_smooth(inner_list, (cx, cy), angle, beta)

        # Penalty for overlaps (soft)
        penalty = overlap_penalty(inner_list)
        
        return side + 100.0 * penalty

    # Initialize params from best found so far
    params = []
    for i in range(N):
        params.extend(initial_inner[i])
    params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

    # Run optimization with increasing beta (annealing the smoothness)
    betas = [10.0, 30.0, 100.0]
    
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break

        try:
            res = minimize(
                lambda p: objective_joint_smooth(p, beta),
                params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 3000}
            )
            
            # Check if the result is valid and better
            new_inner = []
            for i in range(N):
                new_inner.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
            
            if check_disjoint(new_inner):
                new_side = compute_required_side_fast(new_inner, (res.x[N * 3], res.x[N * 3 + 1]), res.x[N * 3 + 2])
                if new_side < best_side:
                    best_side = new_side
                    best_inner = new_inner
                    best_outer_center = (res.x[N * 3], res.x[N * 3 + 1])
                    best_outer_angle = res.x[N * 3 + 2]
                    params = res.x
        except Exception:
            pass

    # --- Phase 3: Final Polish with Exact Objective ---
    # Run Nelder-Mead on the exact (nonsmooth) objective for final refinement
    def objective_joint_exact(params):
        if time.time() - start_time > DEADLINE - 5:
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

        side = compute_required_side_fast(inner_list, (cx, cy), angle)

        # Penalty for overlaps (hard)
        penalty = overlap_penalty(inner_list, min_dist=SQRT3 - 1e-5)
        
        return side + 1000.0 * penalty

    if time.time() - start_time < DEADLINE - 10:
        try:
            res = minimize(
                objective_joint_exact,
                params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 5000}
            )
            
            new_inner = []
            for i in range(N):
                new_inner.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
            
            if check_disjoint(new_inner):
                new_side = compute_required_side_fast(new_inner, (res.x[N * 3], res.x[N * 3 + 1]), res.x[N * 3 + 2])
                if new_side < best_side:
                    best_side = new_side
                    best_inner = new_inner
                    best_outer_center = (res.x[N * 3], res.x[N * 3 + 1])
                    best_outer_angle = res.x[N * 3 + 2]
        except Exception:
            pass

    # Final verification and cleanup
    # Ensure we return a valid configuration
    if not check_disjoint(best_inner):
        # Fallback to initial if optimization broke it
        best_inner = [list(x) for x in initial_inner]
        best_side = compute_required_side_fast(best_inner, (cx_avg, cy_avg), 0.0)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0

    return best_inner, best_outer_center, best_side, best_outer_angle
