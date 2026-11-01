"""Improved program: softer constraints, multiple restarts, adaptive SA, high-precision polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """Compute minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """Smooth (log-sum-exp) surrogate for the max projection."""
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
    if smooth_max < 0:
        smooth_max = 0.0
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

def overlap_penalty(inner, min_dist=SQRT3 - 0.002):
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
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    # --- Initial hexagonal lattice (11 centers) ---
    axial = [(0, 0),
             (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, -1), (1, 1), (-2, 1), (-1, -1)]
    s0 = 1.0 + 1e-7
    init_centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial]
    init_inner = [[x, y, 0.0] for x, y in init_centers]
    cx0 = sum(c[0] for c in init_centers) / len(init_centers)
    cy0 = sum(c[1] for c in init_centers) / len(init_centers)

    best_side = compute_required_side_fast(init_inner, (cx0, cy0), 0.0)
    best_inner = init_inner
    best_outer_center = (cx0, cy0)
    best_outer_angle = 0.0
    best_params = None

    # ============ Phase 1: Optimize outer hexagon (center + rotation) ============
    def obj_outer_smooth(p):
        return compute_required_side_smooth(init_inner, (p[0], p[1]), p[2], beta=50.0)

    def obj_outer_exact(p):
        return compute_required_side_fast(init_inner, (p[0], p[1]), p[2])

    best_val_p1 = float('inf')
    best_p1 = None
    init_angles = list(range(0, 360, 15))
    for ia in init_angles:
        if time.time() > deadline - 60:
            break
        p0 = np.array([cx0, cy0, float(ia)])
        try:
            r1 = minimize(obj_outer_smooth, p0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 600})
            r2 = minimize(obj_outer_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 1000})
            if r2.fun < best_val_p1:
                best_val_p1 = r2.fun
                best_p1 = r2.x.copy()
        except Exception:
            pass
    if best_p1 is not None:
        best_outer_center = (float(best_p1[0]), float(best_p1[1]))
        best_outer_angle = float(best_p1[2])
        best_side = best_val_p1
    else:
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(init_inner, best_outer_center, best_outer_angle)

    # ============ Phase 2: Joint optimization with SOFT contact constraints ============
    # Contact graph for the 3-4-3 triangular lattice
    contact_edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
        (1, 7), (2, 8), (3, 9), (4, 10),
        (7, 8), (8, 9), (9, 10), (10, 7)
    ]
    target_dist = SQRT3  # Target center distance for touching hexagons

    def soft_contact_penalty(inner):
        """Soft penalty: allow small deviations from target distance."""
        pen = 0.0
        for i, j in contact_edges:
            xi, yi, _ = inner[i]
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            dev = d - target_dist
            # Only penalize if distance is LESS than target (overlap)
            if dev < -0.01:  # Overlap
                pen += (dev + 0.01) ** 2 * 100.0
            elif dev > 0.05:  # Too far apart (wastes space)
                pen += (dev - 0.05) ** 2 * 10.0
        return pen

    def obj_joint_smooth(params, beta=60.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il) + soft_contact_penalty(il)
        return side + 50.0 * pen

    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il) + soft_contact_penalty(il)
        return side + 200.0 * pen

    # Build initial params
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = best_outer_center[0]
    p0[N * 3 + 1] = best_outer_center[1]
    p0[N * 3 + 2] = best_outer_angle

    current = p0.copy()
    current_val = obj_joint_smooth(current)
    best_params = current.copy()
    best_val_j = current_val

    # ============ Simulated Annealing with adaptive step sizes ============
    T = 1.5
    T_min = 0.002
    cool = 0.9997
    max_iter = 15000
    np.random.seed(42)
    
    # Adaptive step sizes
    step_pos = 0.3
    step_angle = 20.0
    step_outer_pos = 0.25
    step_outer_angle = 6.0
    
    acceptance_rate = 1.0
    window = 100
    
    for it in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        nb = current.copy()
        r = np.random.random()
        if r < 0.65:
            # perturb one inner hexagon
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            if dim == 2:
                nb[idx * 3 + 2] += np.random.normal(0, step_angle)
            else:
                nb[idx * 3 + dim] += np.random.normal(0, step_pos)
        elif r < 0.85:
            # perturb outer center
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, step_outer_pos)
        else:
            # perturb outer angle
            nb[N * 3 + 2] += np.random.normal(0, step_outer_angle)

        nb_val = obj_joint_smooth(nb)
        accepted = False
        if nb_val < current_val:
            accepted = True
        else:
            delta = nb_val - current_val
            if delta < 10.0:  # Only consider reasonable uphill moves
                if np.random.random() < math.exp(-delta / max(T, 1e-10)):
                    accepted = True
        
        if accepted:
            current = nb
            current_val = nb_val
            if current_val < best_val_j:
                best_params = current.copy()
                best_val_j = current_val
            acceptance_rate = 0.9 * acceptance_rate + 0.1 * 1.0
        else:
            acceptance_rate = 0.9 * acceptance_rate + 0.1 * 0.0
        
        # Adaptive step size adjustment
        if it % window == 0 and it > 0:
            if acceptance_rate > 0.7:
                step_pos *= 1.2
                step_angle *= 1.2
                step_outer_pos *= 1.2
                step_outer_angle *= 1.2
            elif acceptance_rate < 0.3:
                step_pos *= 0.8
                step_angle *= 0.8
                step_outer_pos *= 0.8
                step_outer_angle *= 0.8
            acceptance_rate = 0.5
        
        T *= cool
        if T < T_min:
            break

    # Extract best joint solution
    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])

    # Verify with SAT and compute exact side
    if check_disjoint(best_inner_j, eps=1e-5):
        exact_side = compute_required_side_fast(best_inner_j, best_oc_j, best_oa_j)
        if exact_side < best_side:
            best_inner = best_inner_j
            best_outer_center = best_oc_j
            best_outer_angle = best_oa_j
            best_side = exact_side
            best_params = best_params.copy()

    # ============ Phase 3: Multiple restarts with perturbed initial conditions ============
    if time.time() < deadline - 25:
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            # Perturb the best solution found so far
            p_restart = best_params.copy() if best_params is not None else p0.copy()
            np.random.seed(100 + restart * 137)
            for i in range(N):
                p_restart[i * 3] += np.random.normal(0, 0.1)
                p_restart[i * 3 + 1] += np.random.normal(0, 0.1)
                p_restart[i * 3 + 2] += np.random.normal(0, 10.0)
            p_restart[N * 3] += np.random.normal(0, 0.1)
            p_restart[N * 3 + 1] += np.random.normal(0, 0.1)
            p_restart[N * 3 + 2] += np.random.normal(0, 5.0)
            
            # Quick SA from this restart
            T_r = 0.5
            T_min_r = 0.005
            cool_r = 0.9995
            max_iter_r = 3000
            current_r = p_restart.copy()
            current_val_r = obj_joint_smooth(current_r)
            best_r = current_r.copy()
            best_val_r = current_val_r
            
            step_pos_r = 0.2
            step_angle_r = 15.0
            step_outer_pos_r = 0.15
            step_outer_angle_r = 4.0
            
            for it_r in range(max_iter_r):
                if time.time() > deadline - 15:
                    break
                
                nb_r = current_r.copy()
                r_r = np.random.random()
                if r_r < 0.65:
                    idx_r = np.random.randint(0, N)
                    dim_r = np.random.randint(0, 3)
                    if dim_r == 2:
                        nb_r[idx_r * 3 + 2] += np.random.normal(0, step_angle_r)
                    else:
                        nb_r[idx_r * 3 + dim_r] += np.random.normal(0, step_pos_r)
                elif r_r < 0.85:
                    dim_r = np.random.randint(0, 2)
                    nb_r[N * 3 + dim_r] += np.random.normal(0, step_outer_pos_r)
                else:
                    nb_r[N * 3 + 2] += np.random.normal(0, step_outer_angle_r)

                nb_val_r = obj_joint_smooth(nb_r)
                if nb_val_r < current_val_r or np.random.random() < math.exp(-(nb_val_r - current_val_r) / max(T_r, 1e-10)):
                    current_r = nb_r
                    current_val_r = nb_val_r
                    if current_val_r < best_val_r:
                        best_r = current_r.copy()
                        best_val_r = current_val_r
                T_r *= cool_r
                if T_r < T_min_r:
                    break
            
            # Check if this restart improved
            if best_val_r < best_val_j:
                best_params = best_r.copy()
                best_val_j = best_val_r
                best_inner_j = []
                for i in range(N):
                    best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
                best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
                best_oa_j = float(best_params[N * 3 + 2])
                
                if check_disjoint(best_inner_j, eps=1e-5):
                    exact_side = compute_required_side_fast(best_inner_j, best_oc_j, best_oa_j)
                    if exact_side < best_side:
                        best_inner = best_inner_j
                        best_outer_center = best_oc_j
                        best_outer_angle = best_oa_j
                        best_side = exact_side

    # ============ Phase 4: Final high-precision polish ============
    if time.time() < deadline - 5 and best_params is not None:
        def obj_polish(params):
            il = []
            for i in range(N):
                il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
            ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
            side = compute_required_side_fast(il, (ocx, ocy), oa)
            pen = overlap_penalty(il) + soft_contact_penalty(il)
            return side + 1000.0 * pen

        try:
            res = minimize(obj_polish, best_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000})
            if res.fun < best_val_j:
                polish_inner = []
                for i in range(N):
                    polish_inner.append([float(res.x[i * 3]), float(res.x[i * 3 + 1]), float(res.x[i * 3 + 2])])
                polish_oc = (float(res.x[N * 3]), float(res.x[N * 3 + 1]))
                polish_oa = float(res.x[N * 3 + 2])
                if check_disjoint(polish_inner, eps=1e-6):
                    polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
                    if polish_side < best_side:
                        best_inner = polish_inner
                        best_outer_center = polish_oc
                        best_outer_angle = polish_oa
                        best_side = polish_side
        except Exception:
            pass

    # Final SAT verification
    if not check_disjoint(best_inner, eps=1e-5):
        # Fallback to initial lattice if somehow invalid
        best_inner = init_inner
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
