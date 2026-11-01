"""Discrete topology enumeration + support-function Minkowski-sum enclosing hexagon,
with per-hexagon discrete orientation {0,30,60} and fine continuous polish."""
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

def hex_vertices_local(side=1.0, angle_rad=0.0):
    """Return 6 vertices of a unit hexagon centered at origin with given rotation."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([side * np.cos(thetas), side * np.sin(thetas)])

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
    """Smooth surrogate for the max projection."""
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
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < SQRT3 * 1.5 and d > 0:
                pen += 0.01 * (SQRT3 * 1.5 - d) ** 2
    return pen

def axial_to_xy(q, r, s=1.0):
    return (1.5 * s * q, SQRT3 * s * (r + q / 2.0))

def enumerate_polyhexes(n_cells):
    """
    Enumerate all free polyhexes of size n_cells via BFS on axial coordinates.
    Returns list of sets of (q, r) tuples, canonicalized.
    """
    if n_cells == 1:
        return [frozenset({(0, 0)})]
    
    neighbors = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    # Start from all polyhexes of size n-1
    prev = enumerate_polyhexes(n_cells - 1)
    seen = set()
    result = []
    
    for ph in prev:
        # Try adding each neighbor cell to each cell of the polyhex
        for (q, r) in ph:
            for dq, dr in neighbors:
                nq, nr = q + dq, r + dr
                if (nq, nr) in ph:
                    continue
                new_ph = frozenset(ph | {(nq, nr)})
                # Canonicalize: translate so min q is 0, then min r is 0
                min_q = min(c[0] for c in new_ph)
                min_r = min(c[1] for c in new_ph)
                canon = frozenset((c[0] - min_q, c[1] - min_r) for c in new_ph)
                
                # Also check rotations/reflections for free polyhexes
                # For now, just use translated canonical form (one-sided)
                # We'll deduplicate by trying all 6 rotations
                variants = set()
                for rot in range(6):
                    rotated = set()
                    for (cq, cr) in canon:
                        # Rotate in axial coordinates
                        if rot == 0:
                            rotated.add((cq, cr))
                        elif rot == 1:
                            rotated.add((-cr, cq + cr))
                        elif rot == 2:
                            rotated.add((-cq - cr, cq))
                        elif rot == 3:
                            rotated.add((-cq, -cr))
                        elif rot == 4:
                            rotated.add((cr, -cq - cr))
                        elif rot == 5:
                            rotated.add((cq + cr, -cq))
                    # Normalize
                    mq = min(x[0] for x in rotated)
                    mr = min(x[1] for x in rotated)
                    variants.add(frozenset((x[0] - mq, x[1] - mr) for x in rotated))
                
                if variants not in seen:
                    seen.add(variants)
                    # Use the lexicographically smallest variant as representative
                    rep = min(variants, key=lambda v: sorted(v))
                    result.append(set(rep))
    
    return result

def build_config_from_polyhex(polyhex, s=1.0 + 1e-8):
    """Build inner hexagon positions from a polyhex (set of axial coords)."""
    centers = [axial_to_xy(q, r, s) for q, r in sorted(polyhex)]
    return [[x, y, 0.0] for x, y in centers]

def discrete_orientation_search(inner_base, outer_center, outer_angle_deg, deadline):
    """
    For a given base configuration, try all combinations of discrete orientations
    {0, 30, 60} degrees per hexagon and find the one minimizing the enclosing side.
    Uses a greedy/beam approach to avoid 3^11 = 177147 full evaluations.
    """
    if time.time() > deadline - 5:
        return inner_base, outer_center, outer_angle_deg, compute_required_side_fast(inner_base, outer_center, outer_angle_deg)
    
    best_side = compute_required_side_fast(inner_base, outer_center, outer_angle_deg)
    best_inner = [list(x) for x in inner_base]
    
    # Greedy: for each hexagon, try the 3 orientations and keep the best
    # Iterate a few times for convergence
    orientations = [0.0, 30.0, 60.0]
    for iteration in range(3):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            orig_angle = best_inner[i][2]
            for new_angle in orientations:
                trial = [list(x) for x in best_inner]
                trial[i][2] = new_angle
                if not check_disjoint(trial, eps=1e-5):
                    continue
                side = compute_required_side_fast(trial, outer_center, outer_angle_deg)
                if side < best_side - 1e-9:
                    best_side = side
                    best_inner = trial
                    improved = True
        if not improved:
            break
    
    return best_inner, outer_center, outer_angle_deg, best_side

def optimize_composite_rigid(axial_coords, deadline):
    """
    Rigid body optimization: global rotation, translation, outer center, outer angle.
    """
    s0 = 1.0 + 1e-8
    centers = [axial_to_xy(q, r, s0) for q, r in axial_coords]
    base_positions = [[x, y, 0.0] for x, y in centers]
    
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)
    
    def obj_rigid(p):
        comp_angle = p[0]
        dx, dy = p[1], p[2]
        ox, oy = p[3], p[4]
        outer_angle = p[5]
        
        ca = math.cos(math.radians(comp_angle))
        sa = math.sin(math.radians(comp_angle))
        inner = []
        for i in range(N):
            bx, by, ba = base_positions[i]
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            na = ba + comp_angle
            inner.append([nx, ny, na])
        
        return compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=60.0)
    
    def obj_rigid_exact(p):
        comp_angle = p[0]
        dx, dy = p[1], p[2]
        ox, oy = p[3], p[4]
        outer_angle = p[5]
        ca = math.cos(math.radians(comp_angle))
        sa = math.sin(math.radians(comp_angle))
        inner = []
        for i in range(N):
            bx, by, ba = base_positions[i]
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            na = ba + comp_angle
            inner.append([nx, ny, na])
        return compute_required_side_fast(inner, (ox, oy), outer_angle)
    
    best_val = float('inf')
    best_p = None
    
    for comp_a in range(0, 60, 3):
        for outer_a in range(0, 60, 5):
            if time.time() > deadline - 15:
                break
            p0 = np.array([float(comp_a), 0.0, 0.0, cx0, cy0, float(outer_a)])
            try:
                r1 = minimize(obj_rigid, p0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400})
                r2 = minimize(obj_rigid_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 600})
                if r2.fun < best_val:
                    best_val = r2.fun
                    best_p = r2.x.copy()
            except Exception:
                pass
    
    if best_p is None:
        inner = [[x, y, 0.0] for x, y in centers]
        return inner, (cx0, cy0), 0.0, compute_required_side_fast(inner, (cx0, cy0), 0.0)
    
    comp_angle = float(best_p[0])
    dx, dy = float(best_p[1]), float(best_p[2])
    ox, oy = float(best_p[3]), float(best_p[4])
    outer_angle = float(best_p[5])
    
    ca = math.cos(math.radians(comp_angle))
    sa = math.sin(math.radians(comp_angle))
    inner = []
    for i in range(N):
        bx, by, ba = base_positions[i]
        rx = bx - cx0
        ry = by - cy0
        nx = cx0 + ca * rx - sa * ry + dx
        ny = cy0 + sa * rx + ca * ry + dy
        na = ba + comp_angle
        inner.append([nx, ny, na])
    
    return inner, (ox, oy), outer_angle, best_val

def full_joint_optimize(init_inner, init_oc, init_oa, deadline):
    """Full joint optimization with simulated annealing."""
    def obj_joint_smooth(params, beta=60.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 2000.0 * pen

    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 10000.0 * pen

    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = init_oc[0]
    p0[N * 3 + 1] = init_oc[1]
    p0[N * 3 + 2] = init_oa

    current = p0.copy()
    current_val = obj_joint_smooth(current)
    best_params = current.copy()
    best_val_j = current_val

    T = 1.0
    T_min = 0.001
    cool = 0.9996
    max_iter = 6000
    np.random.seed(123)
    for it in range(max_iter):
        if time.time() > deadline - 35:
            break
        nb = current.copy()
        r = np.random.random()
        scale = max(0.1, T)
        if r < 0.60:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, 0.3 * scale)
        elif r < 0.78:
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, 12.0 * scale)
        elif r < 0.88:
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, 0.2 * scale)
        else:
            nb[N * 3 + 2] += np.random.normal(0, 5.0 * scale)

        nb_val = obj_joint_smooth(nb)
        if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val_j:
                best_params = current.copy()
                best_val_j = current_val
        T *= cool
        if T < T_min:
            break

    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])

    return best_inner_j, best_oc_j, best_oa_j, best_params

def final_polish(params, deadline):
    """High-precision multi-stage polish."""
    def obj_smooth(p, beta=80.0):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen

    def obj_exact(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 20000.0 * pen

    current = params.copy()
    
    try:
        r1 = minimize(obj_smooth, current, method='Nelder-Mead',
                      options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2500})
        current = r1.x
    except Exception:
        pass
    
    if time.time() < deadline - 8:
        try:
            r2 = minimize(obj_exact, current, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 2500})
            current = r2.x
        except Exception:
            pass
    
    if time.time() < deadline - 3:
        cur_val = obj_exact(current)
        step = 0.01
        for _ in range(40):
            if time.time() > deadline - 3:
                break
            improved = False
            for j in range(len(current)):
                for sign in [1, -1]:
                    trial = current.copy()
                    trial[j] += sign * step
                    trial_val = obj_exact(trial)
                    if trial_val < cur_val - 1e-13:
                        current = trial
                        cur_val = trial_val
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-10:
                    break
    
    return current

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    # --- Enumerate polyhex topologies ---
    # For N=11, enumerate all free polyhexes of size 11
    # This is the key structural change: discrete topology enumeration
    try:
        polyhexes = enumerate_polyhexes(N)
    except Exception:
        polyhexes = []
    
    # If enumeration is too slow or fails, fall back to hand-crafted configs
    if not polyhexes or time.time() > deadline - 100:
        polyhexes = []
    
    # Always include hand-crafted configs as fallback
    axial1 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, -1), (1, 1), (-2, 1), (-1, -1)]
    axial2 = [(-1, -1), (0, -1), (1, -1), (2, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (2, 1)]
    axial3 = [(-1, -1), (0, -1), (1, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1)]
    axial4 = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    
    hand_configs = [set(axial1), set(axial2), set(axial3), set(axial4)]
    
    # Combine enumerated and hand-crafted, deduplicate
    all_configs = list(polyhexes) + hand_configs
    seen_shapes = set()
    unique_configs = []
    for cfg in all_configs:
        # Canonicalize for dedup
        min_q = min(c[0] for c in cfg)
        min_r = min(c[1] for c in cfg)
        canon = frozenset((c[0] - min_q, c[1] - min_r) for c in cfg)
        if canon not in seen_shapes:
            seen_shapes.add(canon)
            unique_configs.append(cfg)
    
    # Limit to top K configs by some heuristic (e.g., bounding box area)
    def config_score(cfg):
        xs = [c[0] for c in cfg]
        ys = [c[1] for c in cfg]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    
    unique_configs.sort(key=config_score)
    max_configs = 20
    unique_configs = unique_configs[:max_configs]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1: Rigid composite optimization for each config
    for ci, axial in enumerate(unique_configs):
        if time.time() > deadline - 50:
            break
        inner, oc, oa, side_val = optimize_composite_rigid(axial, deadline)
        
        if check_disjoint(inner, eps=1e-5) and side_val < best_side:
            best_side = side_val
            best_inner = inner
            best_outer_center = oc
            best_outer_angle = oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(inner):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = oc[0]
            p0[N * 3 + 1] = oc[1]
            p0[N * 3 + 2] = oa
            best_params = p0.copy()
    
    # Phase 1b: Discrete orientation search on the best config
    if best_inner is not None and time.time() < deadline - 40:
        d_inner, d_oc, d_oa, d_side = discrete_orientation_search(
            best_inner, best_outer_center, best_outer_angle, deadline)
        if d_side < best_side:
            best_inner = d_inner
            best_outer_center = d_oc
            best_outer_angle = d_oa
            best_side = d_side
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(best_inner):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = best_outer_center[0]
            p0[N * 3 + 1] = best_outer_center[1]
            p0[N * 3 + 2] = best_outer_angle
            best_params = p0.copy()
    
    if best_inner is None:
        axial = unique_configs[0] if unique_configs else axial1
        s0 = 1.0 + 1e-8
        centers = [axial_to_xy(q, r, s0) for q, r in axial]
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = cx0
        p0[N * 3 + 1] = cy0
        p0[N * 3 + 2] = 0.0
        best_params = p0.copy()
    
    # Phase 2: Full joint optimization from best config
    if time.time() < deadline - 40:
        joint_inner, joint_oc, joint_oa, joint_params = full_joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params
    
    # Phase 2b: Discrete orientation refinement after joint optimization
    if time.time() < deadline - 25:
        d_inner, d_oc, d_oa, d_side = discrete_orientation_search(
            best_inner, best_outer_center, best_outer_angle, deadline)
        if d_side < best_side:
            best_inner = d_inner
            best_outer_center = d_oc
            best_outer_angle = d_oa
            best_side = d_side
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(best_inner):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = best_outer_center[0]
            p0[N * 3 + 1] = best_outer_center[1]
            p0[N * 3 + 2] = best_outer_angle
            best_params = p0.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 15 and best_params is not None:
        polished = final_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
        polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
        polish_oa = float(polished[N * 3 + 2])
        
        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side
    
    # Phase 4: Fine outer parameter sweep
    if time.time() < deadline - 8 and best_params is not None:
        cur_inner = best_inner
        cur_oc = best_outer_center
        cur_oa = best_outer_angle
        cur_side = best_side
        
        for da in [1, -1, 2, -2, 0.5, -0.5]:
            if time.time() > deadline - 5:
                break
            new_oa = cur_oa + da
            new_side = compute_required_side_fast(cur_inner, cur_oc, new_oa)
            if new_side < cur_side - 1e-9:
                cur_oa = new_oa
                cur_side = new_side
        
        for dx in [0.01, -0.01, 0.02, -0.02]:
            for dy in [0.01, -0.01, 0.02, -0.02]:
                if time.time() > deadline - 5:
                    break
                new_oc = (cur_oc[0] + dx, cur_oc[1] + dy)
                new_side = compute_required_side_fast(cur_inner, new_oc, cur_oa)
                if new_side < cur_side - 1e-9:
                    cur_oc = new_oc
                    cur_side = new_side
        
        if cur_side < best_side:
            best_outer_center = cur_oc
            best_outer_angle = cur_oa
            best_side = cur_side
    
    # Final verification
    if not check_disjoint(best_inner, eps=1e-5):
        s0 = 1.0 + 1e-8
        centers = [axial_to_xy(q, r, s0) for q, r in axial1]
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
