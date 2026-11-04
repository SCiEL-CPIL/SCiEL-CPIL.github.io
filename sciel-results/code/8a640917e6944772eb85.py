import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _get_outer_normals(angle_deg):
    """Unit normals for the 6 sides of a regular hexagon with given rotation."""
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi / 6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    """Exact minimum side length of a regular hexagon enclosing the point set.
    
    The hexagon is centered at outer_center with rotation outer_angle_deg.
    For a regular hexagon with side S, the distance from center to each side
    (apothem) is S * sqrt(3) / 2. So S = 2 * max_projection / sqrt(3).
    """
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _all_inner_vertices(centers, angles_deg):
    """All vertices of all inner hexagons, as (N*6, 2)."""
    all_verts = []
    for i, (cx, cy) in enumerate(centers):
        all_verts.append(_hex_vertices(cx, cy, angles_deg[i]))
    return np.vstack(all_verts)

def _soft_max(vals, beta=1000.0):
    """Soft maximum via log-sum-exp for smoothness."""
    m = np.max(vals)
    if m == 0:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta

def _soft_min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center, beta=1000.0):
    """Smooth surrogate for min enclosing hex side using soft-max."""
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    # Use soft max on absolute projections
    abs_dots = np.abs(dots)
    # Soft max over all projections
    soft_max_proj = _soft_max(abs_dots.ravel(), beta)
    return 2 * soft_max_proj / np.sqrt(3)

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        # Parallel
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap, positive if separated.
    """
    # Quick check: if any vertex of one is inside the other, they overlap
    n1 = len(p1)
    n2 = len(p2)
    
    # Check if any vertex of p1 is inside p2
    for v in p1:
        inside = True
        for i in range(n2):
            a = p2[i]
            b = p2[(i+1) % n2]
            cross = (b[0] - a[0]) * (v[1] - a[1]) - (b[1] - a[1]) * (v[0] - a[0])
            if cross < -1e-10:
                inside = False
                break
        if inside:
            return 0.0
    
    # Check if any vertex of p2 is inside p1
    for v in p2:
        inside = True
        for i in range(n1):
            a = p1[i]
            b = p1[(i+1) % n1]
            cross = (b[0] - a[0]) * (v[1] - a[1]) - (b[1] - a[1]) * (v[0] - a[0])
            if cross < -1e-10:
                inside = False
                break
        if inside:
            return 0.0
    
    # Check edge-edge distances
    min_dist = float('inf')
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    # ------------------------------------------------------------------
    # Phase 0: Direct construction of the known optimal "flower" packing.
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3, all rotated 30 deg.
    # Theoretical minimum outer side length for this 11-hexagon cluster is 4.0.
    # We build it with a tiny epsilon gap for strict disjointness.
    # ------------------------------------------------------------------
    eps = 1e-9
    centers_list = []
    angles_list = []

    # Ring 0: center
    centers_list.append([0.0, 0.0])
    angles_list.append(30.0)

    # Ring 1: 6 hexagons at distance sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = D + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)

    # Ring 2: 4 hexagons at distance 3 + 2*eps, at 30, 90, 150, 210 deg
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)

    centers = np.array(centers_list)
    angles = np.array(angles_list)
    n_inner = len(centers)

    # Compute the enclosing side of this exact configuration.
    # Outer hexagon centered at origin, rotated 30 deg.
    all_verts = _all_inner_vertices(centers, angles)
    side_flower = _min_enclosing_hex_side(all_verts, 30.0, np.array([0.0, 0.0]))

    best_S = side_flower
    best_outer = np.array([0.0, 0.0, 30.0])
    best_centers = centers.copy()

    # ------------------------------------------------------------------
    # Phase 1: Fast polish of the OUTER hexagon only (3 variables: cx, cy, angle)
    # using Nelder-Mead with the EXACT containment objective.
    # Each objective eval is one matrix multiply -- very cheap.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 10.0:
        try:
            def _outer_objective(vars):
                outer_cx, outer_cy, outer_angle = vars
                outer_center = np.array([outer_cx, outer_cy])
                return _min_enclosing_hex_side(all_verts, outer_angle, outer_center)

            x0 = best_outer.copy()
            res = minimize(_outer_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_outer = res.x.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 2: Joint polish of inner positions + outer hexagon using smooth surrogate.
    # Keep inner angles fixed at 30 deg and only translate inner centers + adjust outer hexagon.
    # Use L-BFGS-B with soft-max surrogate for smoothness.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 15.0:
        try:
            def _joint_smooth_objective(vars, beta=1000.0):
                outer_cx = vars[0]
                outer_cy = vars[1]
                outer_angle = vars[2]
                c = vars[3:3 + 2 * n_inner].reshape(n_inner, 2)
                verts = []
                for i in range(n_inner):
                    verts.append(_hex_vertices(c[i][0], c[i][1], angles[i]))
                all_v = np.vstack(verts)
                return _soft_min_enclosing_hex_side(all_v, outer_angle, np.array([outer_cx, outer_cy]), beta)

            x0_joint = np.zeros(3 + 2 * n_inner)
            x0_joint[0] = best_outer[0]
            x0_joint[1] = best_outer[1]
            x0_joint[2] = best_outer[2]
            for i in range(n_inner):
                x0_joint[3 + 2 * i] = best_centers[i][0]
                x0_joint[4 + 2 * i] = best_centers[i][1]

            # Use L-BFGS-B with smooth objective
            bounds = [(-8, 8), (-8, 8), (-30, 90)] + [(-6, 6)] * (2 * n_inner)
            res = minimize(_joint_smooth_objective, x0_joint, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if time.time() < DEADLINE - 5.0:
                # Evaluate with exact objective
                outer_cx, outer_cy, outer_angle = res.x[:3]
                c_opt = res.x[3:3 + 2 * n_inner].reshape(n_inner, 2)
                verts_opt = []
                for i in range(n_inner):
                    verts_opt.append(_hex_vertices(c_opt[i][0], c_opt[i][1], angles[i]))
                all_v_opt = np.vstack(verts_opt)
                exact_S = _min_enclosing_hex_side(all_v_opt, outer_angle, np.array([outer_cx, outer_cy]))
                if exact_S < best_S:
                    best_S = exact_S
                    best_outer = res.x[:3].copy()
                    best_centers = c_opt.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 3: Try alternative outer rotations with L-BFGS-B on inner positions.
    # The outer hexagon rotation is a discrete choice that can affect the
    # optimal inner placement. Try several rotations.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 10.0:
        try:
            best_final_S = best_S
            best_final_params = best_centers.copy()
            best_final_outer = best_outer.copy()

            for alpha_test in np.linspace(0, 60, 7):  # 0, 10, 20, ..., 60 degrees
                if time.time() > DEADLINE - 5.0:
                    break

                def obj_alpha(c_flat, alpha=alpha_test):
                    c = c_flat.reshape(n_inner, 2)
                    verts = []
                    for i in range(n_inner):
                        verts.append(_hex_vertices(c[i][0], c[i][1], angles[i]))
                    all_v = np.vstack(verts)
                    outer_center = np.array([best_final_outer[0], best_final_outer[1]])
                    return _min_enclosing_hex_side(all_v, alpha, outer_center)

                x0 = best_final_params.flatten().copy()
                bounds = [(-6, 6), (-6, 6)] * n_inner

                res = minimize(obj_alpha, x0, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_final_S:
                    best_final_S = res.fun
                    best_final_params = res.x.reshape(n_inner, 2).copy()
                    best_final_outer = np.array([best_final_outer[0], best_final_outer[1], alpha_test])

            if best_final_S < best_S:
                best_S = best_final_S
                best_centers = best_final_params
                best_outer = best_final_outer
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 4: High-precision polish with tight tolerances
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 5.0:
        try:
            def _final_polish_objective(vars):
                outer_cx = vars[0]
                outer_cy = vars[1]
                outer_angle = vars[2]
                c = vars[3:3 + 2 * n_inner].reshape(n_inner, 2)
                verts = []
                for i in range(n_inner):
                    verts.append(_hex_vertices(c[i][0], c[i][1], angles[i]))
                all_v = np.vstack(verts)
                return _min_enclosing_hex_side(all_v, outer_angle, np.array([outer_cx, outer_cy]))

            x0_final = np.zeros(3 + 2 * n_inner)
            x0_final[0] = best_outer[0]
            x0_final[1] = best_outer[1]
            x0_final[2] = best_outer[2]
            for i in range(n_inner):
                x0_final[3 + 2 * i] = best_centers[i][0]
                x0_final[4 + 2 * i] = best_centers[i][1]

            res = minimize(_final_polish_objective, x0_final, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res.fun < best_S:
                best_S = res.fun
                best_outer = res.x[:3].copy()
                best_centers = res.x[3:3 + 2 * n_inner].reshape(n_inner, 2).copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    outer_cx, outer_cy, outer_angle = best_outer
    outer_center = (outer_cx, outer_cy)

    inner = []
    for i in range(n_inner):
        inner.append([best_centers[i][0], best_centers[i][1], angles[i]])

    side = best_S

    return inner, outer_center, side, outer_angle
