"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
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
    """
    Smooth (log-sum-exp) surrogate for the max absolute projection.
    Uses both positive and negative projections to properly handle containment.
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
    # Use absolute values for proper containment check
    P_abs = np.abs(P)
    m = P_abs.max()
    if m < -1e6:
        return 0.0
    # Center around max for numerical stability
    P_centered = P_abs - m
    smooth_max = m + math.log(np.sum(np.exp(beta * P_centered))) / beta
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
    t0 = time.time()
    deadline = t0 + 142.0  # keep 8 s safety margin

    # --- Multiple initial hexagonal lattice configurations ---
    # Flower: 1 center + 6 ring + 4 outer
    axial_flower = [(0, 0),
                    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                    (2, -1), (1, 1), (-2, 1), (-1, -1)]
    
    # Row-based 4-3-4
    axial_row434 = []
    # Row -1: 4 hexagons
    for i in range(4):
        axial_row434.append((i - 1, -1))
    # Row 0: 3 hexagons
    for i in range(3):
        axial_row434.append((i, 0))
    # Row 1: 4 hexagons
    for i in range(4):
        axial_row434.append((i - 1, 1))
    
    # 3-5-3
    axial_353 = []
    # Row -1: 3
    for i in range(3):
        axial_353.append((i - 1, -1))
    # Row 0: 5
    for i in range(5):
        axial_353.append((i - 2, 0))
    # Row 1: 3
    for i in range(3):
        axial_353.append((i - 1, 1))
    
    # 5-3-3 variant
    axial_533 = []
    # Row -1: 5
    for i in range(5):
        axial_533.append((i - 2, -1))
    # Row 0: 3
    for i in range(3):
        axial_533.append((i, 0))
    # Row 1: 3
    for i in range(3):
        axial_533.append((i - 1, 1))
    
    # Compact 2-3-2-3-1 (staggered rows)
    axial_23231 = []
    # Row -2: 2
    for i in range(2):
        axial_23231.append((i, -2))
    # Row -1: 3
    for i in range(3):
        axial_23231.append((i - 1, -1))
    # Row 0: 2
    for i in range(2):
        axial_23231.append((i - 1, 0))
    # Row 1: 3
    for i in range(3):
        axial_23231.append((i - 1, 1))
    # Row 2: 1
    axial_23231.append((0, 2))
    
    # Diamond: 1-2-3-2-1
    axial_diamond = []
    axial_diamond.append((0, -2))
    for i in range(2):
        axial_diamond.append((i - 1, -1))
    for i in range(3):
        axial_diamond.append((i - 1, 0))
    for i in range(2):
        axial_diamond.append((i - 1, 1))
    axial_diamond.append((0, 2))
    
    # Additional configurations for better exploration
    # Hexagonal ring with center
    axial_ring = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                  (2, -1), (1, 1), (-2, 1), (-1, -1), (2, 0), (-2, 0)]
    # Trim to 11
    axial_ring = axial_ring[:11]
    
    # Linear chain with branches
    axial_chain = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                   (1, 1), (2, 1), (3, 1),
                   (1, -1), (2, -1), (3, -1)]
    
    # Compact square-like
    axial_square = [(0, 0), (1, 0), (2, 0),
                    (0, 1), (1, 1), (2, 1),
                    (0, -1), (1, -1), (2, -1),
                    (1, 2), (1, -2)]
    
    configs = [axial_flower, axial_row434, axial_353, axial_533, axial_23231, axial_diamond,
               axial_ring, axial_chain, axial_square]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_params = None
    
    np.random.seed(42)
    
    for cfg_idx, axial in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        s0 = 1.0 + 1e-7
        init_centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial]
        init_inner = [[x, y, 0.0] for x, y in init_centers]
        cx0 = sum(c[0] for c in init_centers) / len(init_centers)
        cy0 = sum(c[1] for c in init_centers) / len(init_centers)
        
        # ============ Phase 1: Optimize outer hexagon (center + rotation) ============
        def obj_outer_smooth(p, inner=init_inner):
            return compute_required_side_smooth(inner, (p[0], p[1]), p[2], beta=50.0)
        
        def obj_outer_exact(p, inner=init_inner):
            return compute_required_side_fast(inner, (p[0], p[1]), p[2])
        
        best_val_p1 = float('inf')
        best_p1 = None
        init_angles = list(range(0, 360, 30))
        for ia in init_angles:
            if time.time() > deadline - 35:
                break
            p0 = np.array([cx0, cy0, float(ia)])
            try:
                r1 = minimize(obj_outer_smooth, p0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 400})
                r2 = minimize(obj_outer_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 800})
                if r2.fun < best_val_p1:
                    best_val_p1 = r2.fun
                    best_p1 = r2.x.copy()
            except Exception:
                pass
        if best_p1 is not None:
            ocx1, ocy1, oa1 = float(best_p1[0]), float(best_p1[1]), float(best_p1[2])
        else:
            ocx1, ocy1, oa1 = cx0, cy0, 0.0
        
        # ============ Phase 2: Joint optimization (inner positions + outer) ============
        def obj_joint_smooth(params, inner_cfg=init_inner, beta=60.0):
            il = []
            for i in range(N):
                il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
            ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
            side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
            pen = overlap_penalty(il)
            return side + 400.0 * pen
        
        def obj_joint_exact(params, inner_cfg=init_inner):
            il = []
            for i in range(N):
                il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
            ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
            side = compute_required_side_fast(il, (ocx, ocy), oa)
            pen = overlap_penalty(il)
            return side + 2000.0 * pen
        
        # Build initial params
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(init_inner):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = ocx1
        p0[N * 3 + 1] = ocy1
        p0[N * 3 + 2] = oa1
        
        current = p0.copy()
        current_val = obj_joint_smooth(current)
        best_params_cfg = current.copy()
        best_val_j = current_val
        
        # Simulated annealing with adaptive step sizes
        T = 2.0
        T_min = 0.0005
        cool = 0.9993
        max_iter = 15000
        step_scale = 1.0
        for it in range(max_iter):
            if time.time() > deadline - 40:
                break
            nb = current.copy()
            r = np.random.random()
            if r < 0.65:
                idx = np.random.randint(0, N)
                dim = np.random.randint(0, 3)
                if dim == 2:
                    nb[idx * 3 + 2] += np.random.normal(0, 18.0 * step_scale)
                else:
                    nb[idx * 3 + dim] += np.random.normal(0, 0.4 * step_scale)
            elif r < 0.80:
                dim = np.random.randint(0, 2)
                nb[N * 3 + dim] += np.random.normal(0, 0.3 * step_scale)
            else:
                nb[N * 3 + 2] += np.random.normal(0, 8.0 * step_scale)
            
            nb_val = obj_joint_smooth(nb)
            if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
                current = nb
                current_val = nb_val
                if current_val < best_val_j:
                    best_params_cfg = current.copy()
                    best_val_j = current_val
                    step_scale = max(0.5, step_scale * 0.95)
            else:
                step_scale = min(2.0, step_scale * 1.02)
            T *= cool
            if T < T_min:
                break
        
        # Extract best joint solution for this config
        best_inner_cfg = []
        for i in range(N):
            best_inner_cfg.append([float(best_params_cfg[i * 3]), float(best_params_cfg[i * 3 + 1]), float(best_params_cfg[i * 3 + 2])])
        best_oc_cfg = (float(best_params_cfg[N * 3]), float(best_params_cfg[N * 3 + 1]))
        best_oa_cfg = float(best_params_cfg[N * 3 + 2])
        
        # Verify with SAT and compute exact side
        if check_disjoint(best_inner_cfg, eps=1e-5):
            exact_side = compute_required_side_fast(best_inner_cfg, best_oc_cfg, best_oa_cfg)
            if exact_side < best_side:
                best_inner = best_inner_cfg
                best_outer_center = best_oc_cfg
                best_outer_angle = best_oa_cfg
                best_side = exact_side
                best_params = best_params_cfg.copy()
        
        # ============ Phase 3: Final polish with Nelder-Mead on exact objective ============
        if time.time() < deadline - 15:
            def obj_polish(params):
                il = []
                for i in range(N):
                    il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
                ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
                side = compute_required_side_fast(il, (ocx, ocy), oa)
                pen = overlap_penalty(il)
                return side + 5000.0 * pen
            
            try:
                res = minimize(obj_polish, best_params_cfg, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000})
                if res.fun < best_val_j:
                    polish_inner = []
                    for i in range(N):
                        polish_inner.append([float(res.x[i * 3]), float(res.x[i * 3 + 1]), float(res.x[i * 3 + 2])])
                    polish_oc = (float(res.x[N * 3]), float(res.x[N * 3 + 1]))
                    polish_oa = float(res.x[N * 3 + 2])
                    if check_disjoint(polish_inner, eps=1e-5):
                        polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
                        if polish_side < best_side:
                            best_inner = polish_inner
                            best_outer_center = polish_oc
                            best_outer_angle = polish_oa
                            best_side = polish_side
                            best_params = res.x.copy()
            except Exception:
                pass
    
    # Final SAT verification
    if best_inner is None or not check_disjoint(best_inner, eps=1e-5):
        # Fallback to initial flower lattice
        axial = axial_flower
        s0 = 1.0 + 1e-7
        init_centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial]
        best_inner = [[x, y, 0.0] for x, y in init_centers]
        cx0 = sum(c[0] for c in init_centers) / len(init_centers)
        cy0 = sum(c[1] for c in init_centers) / len(init_centers)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    
    # Final high-precision polish on the best solution found
    if best_inner is not None and time.time() < deadline - 5:
        def obj_final(params):
            il = []
            for i in range(N):
                il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
            ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
            side = compute_required_side_fast(il, (ocx, ocy), oa)
            pen = overlap_penalty(il)
            return side + 10000.0 * pen
        
        try:
            p_start = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(best_inner):
                p_start[i * 3] = x
                p_start[i * 3 + 1] = y
                p_start[i * 3 + 2] = a
            p_start[N * 3] = best_outer_center[0]
            p_start[N * 3 + 1] = best_outer_center[1]
            p_start[N * 3 + 2] = best_outer_angle
            
            # Try multiple final polish attempts
            final_candidates = [p_start.copy()]
            for _ in range(2):
                perturbed = p_start.copy()
                perturbed[:N*3] += np.random.normal(0, 0.005, N*3)
                final_candidates.append(perturbed)
            
            for fc in final_candidates:
                if time.time() > deadline - 2:
                    break
                try:
                    res_final = minimize(obj_final, fc, method='Nelder-Mead',
                                         options={'maxiter': 3500, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 4500})
                    if res_final.fun < best_side:
                        final_inner = []
                        for i in range(N):
                            final_inner.append([float(res_final.x[i * 3]), float(res_final.x[i * 3 + 1]), float(res_final.x[i * 3 + 2])])
                        final_oc = (float(res_final.x[N * 3]), float(res_final.x[N * 3 + 1]))
                        final_oa = float(res_final.x[N * 3 + 2])
                        if check_disjoint(final_inner, eps=1e-6):
                            final_side = compute_required_side_fast(final_inner, final_oc, final_oa)
                            if final_side < best_side:
                                best_inner = final_inner
                                best_outer_center = final_oc
                                best_outer_angle = final_oa
                                best_side = final_side
                except Exception:
                    pass
        except Exception:
            pass
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
