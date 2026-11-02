"""Rigid-body composite packing with improved smooth surrogate and contact-aware optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

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
    return np.column_stack([side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    # Normals for the 6 faces of the outer hexagon (pointing outward)
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    # Rotation matrix to transform to outer hexagon's frame
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        # Transform to outer frame
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
    # So s = 2 * apothem / sqrt(3) = 2 * max_proj / sqrt(3)
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
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
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
    centered = P - m
    smooth_max = m + math.log(np.sum(np.exp(beta * centered))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y) tuples or numpy arrays)."""
    # Convert to list of tuples if numpy
    if isinstance(v1, np.ndarray):
        v1 = [tuple(row) for row in v1]
    if isinstance(v2, np.ndarray):
        v2 = [tuple(row) for row in v2]
    
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

def build_lattice_config(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coordinates with spacing s."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def compute_contact_graph(inner):
    """Identify pairs of hexagons that are in contact (distance < threshold)."""
    n = len(inner)
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, _ = inner[i]
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < SQRT3 * 1.05:  # Slightly above touching distance
                contacts.append((i, j, d))
    return contacts

def optimize_composite_rigid(axial_coords, deadline):
    """
    Treat the 11 hexagons as a rigid body, optimize global rotation and outer params.
    """
    s0 = 1.0 + 1e-8
    centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial_coords]
    base_positions = [[x, y, 0.0] for x, y in centers]
    
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)
    
    def obj_rigid(p):
        """p = [comp_angle, dx, dy, ox, oy, outer_angle]"""
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
        
        return compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=50.0)
    
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
    
    # Multi-start over composite angle and outer angle with finer grid
    for comp_a in range(0, 60, 3):
        for outer_a in range(0, 60, 5):
            if time.time() > deadline - 25:
                break
            p0 = np.array([float(comp_a), 0.0, 0.0, cx0, cy0, float(outer_a)])
            try:
                r1 = minimize(obj_rigid, p0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400})
                r2 = minimize(obj_rigid_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 800})
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

def semi_rigid_optimize(axial_coords, deadline):
    """
    Semi-rigid: individual angle offsets with fixed relative positions.
    """
    s0 = 1.0 + 1e-8
    centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial_coords]
    base_positions = [[x, y, 0.0] for x, y in centers]
    
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)
    
    def obj_semi(p):
        angles = p[:N]
        ox, oy, outer_angle = p[N], p[N+1], p[N+2]
        inner = []
        for i in range(N):
            bx, by, _ = base_positions[i]
            inner.append([bx, by, angles[i]])
        return compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=40.0)
    
    def obj_semi_exact(p):
        angles = p[:N]
        ox, oy, outer_angle = p[N], p[N+1], p[N+2]
        inner = []
        for i in range(N):
            bx, by, _ = base_positions[i]
            inner.append([bx, by, angles[i]])
        return compute_required_side_fast(inner, (ox, oy), outer_angle)
    
    best_val = float('inf')
    best_p = None
    
    # More diverse initial conditions
    for outer_a in [0, 15, 30, 45]:
        if time.time() > deadline - 20:
            break
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            p0 = np.zeros(N + 3)
            p0[N] = cx0
            p0[N+1] = cy0
            p0[N+2] = float(outer_a)
            np.random.seed(outer_a * 100 + trial)
            p0[:N] = np.random.uniform(-20, 20, N)
            
            try:
                r1 = minimize(obj_semi, p0, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 600})
                r2 = minimize(obj_semi_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 700, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 1000})
                if r2.fun < best_val:
                    best_val = r2.fun
                    best_p = r2.x.copy()
            except Exception:
                pass
    
    if best_p is None:
        inner = [[x, y, 0.0] for x, y in centers]
        return inner, (cx0, cy0), 0.0, compute_required_side_fast(inner, (cx0, cy0), 0.0)
    
    angles = best_p[:N]
    ox, oy, outer_angle = float(best_p[N]), float(best_p[N+1]), float(best_p[N+2])
    inner = []
    for i in range(N):
        bx, by, _ = base_positions[i]
        inner.append([bx, by, float(angles[i])])
    
    return inner, (ox, oy), outer_angle, best_val

def full_joint_optimize(init_inner, init_oc, init_oa, deadline):
    """Full joint optimization with adaptive simulated annealing and contact-aware moves."""
    def obj_joint_smooth(params, beta=50.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 500.0 * pen

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

    # Compute initial contact graph
    contacts = compute_contact_graph(init_inner)
    
    T = 0.8
    T_min = 0.0005
    cool = 0.9997
    max_iter = 15000
    step_scale = 1.0
    np.random.seed(12345)
    
    for it in range(max_iter):
        if time.time() > deadline - 12:
            break
        nb = current.copy()
        r = np.random.random()
        
        # Contact-aware moves: bias towards moving hexagons that are in contact
        if r < 0.55:
            # Move a hexagon (position)
            if contacts and np.random.random() < 0.7:
                # Pick from contact pairs
                ci, cj, _ = contacts[np.random.randint(0, len(contacts))]
                idx = ci if np.random.random() < 0.5 else cj
            else:
                idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, 0.18 * step_scale)
        elif r < 0.72:
            # Rotate a hexagon
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, 6.0 * step_scale)
        elif r < 0.85:
            # Move outer center
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, 0.10 * step_scale)
        else:
            # Rotate outer
            nb[N * 3 + 2] += np.random.normal(0, 2.5 * step_scale)

        nb_val = obj_joint_smooth(nb)
        delta = nb_val - current_val
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val_j:
                best_params = current.copy()
                best_val_j = current_val
                # Update contact graph periodically
                if it % 500 == 0:
                    il = [[current[i*3], current[i*3+1], current[i*3+2]] for i in range(N)]
                    contacts = compute_contact_graph(il)
            step_scale = min(1.6, step_scale * 1.015)
        else:
            step_scale = max(0.25, step_scale * 0.975)
        
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
    """High-precision Nelder-Mead polish with multiple restarts and tighter tolerances."""
    def obj_polish(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen

    best_params = params.copy()
    best_val = obj_polish(params)
    
    for trial in range(5):
        if time.time() > deadline - 3:
            break
        p_start = params.copy()
        if trial > 0:
            p_start += np.random.normal(0, 0.008, len(params))
        
        try:
            res = minimize(obj_polish, p_start, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 8000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin

    # Define multiple candidate configurations with diverse topologies
    axial1 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (1, 1), (-2, 1), (-1, -1)]
    
    axial2 = [(-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]
    
    axial3 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
              (-1, 1), (0, 1), (1, 1)]
    
    axial4 = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1)]
    
    axial5 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, 0), (0, 2), (-2, 1), (-2, -1)]
    
    # New configurations with different shapes
    axial6 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -2), (1, 1), (-2, 2), (-1, -2)]
    
    axial7 = [(-1, 0), (0, 0), (1, 0), (2, 0),
              (-1, 1), (0, 1), (1, 1),
              (0, 2), (-1, -1), (1, -1), (0, -2)]
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1a: Rigid composite optimization for each config
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 85:
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
    
    # Phase 1b: Semi-rigid optimization for top 3 configs
    for ci, axial in enumerate(configs[:3]):
        if time.time() > deadline - 55:
            break
        inner, oc, oa, side_val = semi_rigid_optimize(axial, deadline)
        
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
    
    if best_inner is None:
        axial = configs[0]
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(axial, s0)
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
    if time.time() < deadline - 30:
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
    
    # Phase 3: Second joint optimization from a different starting point
    if time.time() < deadline - 18 and best_params is not None:
        # Perturb the best solution and run another joint optimization
        np.random.seed(999)
        perturbed = best_params + np.random.normal(0, 0.15, len(best_params))
        pert_inner = [[float(perturbed[i*3]), float(perturbed[i*3+1]), float(perturbed[i*3+2])] for i in range(N)]
        pert_oc = (float(perturbed[N*3]), float(perturbed[N*3+1]))
        pert_oa = float(perturbed[N*3+2])
        
        joint2_inner, joint2_oc, joint2_oa, joint2_params = full_joint_optimize(
            pert_inner, pert_oc, pert_oa, deadline)
        
        if check_disjoint(joint2_inner, eps=1e-5):
            joint2_side = compute_required_side_fast(joint2_inner, joint2_oc, joint2_oa)
            if joint2_side < best_side:
                best_inner = joint2_inner
                best_outer_center = joint2_oc
                best_outer_angle = joint2_oa
                best_side = joint2_side
                best_params = joint2_params
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5 and best_params is not None:
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
    
    # Final verification
    if not check_disjoint(best_inner, eps=1e-5):
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(configs[0], s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
