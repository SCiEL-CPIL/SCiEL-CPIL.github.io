"""
Hexagon packing: 11 unit hexagons in minimal outer hexagon.
Combined approach: PBD-style constraint projection for collision resolution,
plus smooth-surrogate optimization with L-BFGS-B polish.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
SQRT3 = math.sqrt(3)
TWO_PI = 2 * math.pi
APOTHEM = SQRT3 / 2.0
R_HEX = 1.0  # circumradius of unit hexagon

START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0


def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN


def hex_verts_array(cx, cy, side, angle_rad):
    """Vectorized hex vertices: (6,2) array."""
    thetas = angle_rad + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + side * np.cos(thetas),
                            cy + side * np.sin(thetas)])


def required_side(inner, oc, oa):
    """Exact minimum side length for outer hexagon to contain all inner.
    inner: list of (x, y, angle_deg)
    oc: (cx, cy), oa: outer angle in degrees
    Returns side length (scalar)."""
    ox, oy = oc
    base_angles = np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0]) * np.pi / 180.0
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])

    ca = math.cos(math.radians(oa))
    sa = math.sin(math.radians(oa))
    R = np.array([[ca, -sa], [sa, ca]])

    max_proj = 0.0
    for (ix, iy, iang) in inner:
        verts = hex_verts_array(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R
        projs = local @ normals.T
        m = projs.max()
        if m > max_proj:
            max_proj = m
    return 2.0 * max_proj / SQRT3


def required_side_smooth(inner, oc, oa, beta=50.0):
    """Smooth surrogate via log-sum-exp."""
    ox, oy = oc
    base_angles = np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0]) * np.pi / 180.0
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    ca = math.cos(math.radians(oa))
    sa = math.sin(math.radians(oa))
    R = np.array([[ca, -sa], [sa, ca]])

    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_verts_array(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R
        projs = local @ normals.T
        all_projs.append(projs.ravel())
    P = np.concatenate(all_projs)
    m = P.max()
    centered = P - m
    smooth_max = m + math.log(np.sum(np.exp(beta * centered))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3


def sat_overlap(v1, v2, eps=1e-9):
    """Separating axis theorem for two convex polygons."""
    n1, n2 = len(v1), len(v2)
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            nx = -ey
            ny = ex
            ln = math.hypot(nx, ny)
            if ln < 1e-14:
                continue
            nx /= ln
            ny /= ln
            min1 = min(p[0] * nx + p[1] * ny for p in v1)
            max1 = max(p[0] * nx + p[1] * ny for p in v1)
            min2 = min(p[0] * nx + p[1] * ny for p in v2)
            max2 = max(p[0] * nx + p[1] * ny for p in v2)
            if max1 < min2 + eps or max2 < min1 + eps:
                return False
    return True


def check_disjoint(inner, eps=1e-6):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_verts_array(x, y, 1.0, math.radians(a)).tolist())
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True


def overlap_penalty(inner, min_dist=SQRT3 - 0.001):
    """Penalty for center distances below minimum."""
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


def full_objective_smooth(p, beta=50.0):
    """Objective for optimization: smooth side + penalty."""
    inner = []
    for i in range(N):
        inner.append((p[i*3], p[i*3+1], p[i*3+2]))
    ox, oy, oa = p[N*3], p[N*3+1], p[N*3+2]
    side = required_side_smooth(inner, (ox, oy), oa, beta=beta)
    pen = overlap_penalty(inner)
    return side + 1000.0 * pen


def full_objective_exact(p):
    inner = []
    for i in range(N):
        inner.append((p[i*3], p[i*3+1], p[i*3+2]))
    ox, oy, oa = p[N*3], p[N*3+1], p[N*3+2]
    side = required_side(inner, (ox, oy), oa)
    pen = overlap_penalty(inner)
    return side + 10000.0 * pen


def centroid(inner):
    n = len(inner)
    cx = sum(p[0] for p in inner) / n
    cy = sum(p[1] for p in inner) / n
    return cx, cy


# --- PBD-style collision resolution ---

def pbd_resolve_collisions(inner, max_iters=50, tol=1e-8):
    """
    Position-Based Dynamics: iteratively project hexagons apart to resolve overlaps.
    inner: list of [x, y, angle_deg] (mutable)
    Returns modified inner with reduced overlaps.
    """
    n = len(inner)
    for it in range(max_iters):
        if not _time_ok():
            break
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ai = inner[i]
                xj, yj, aj = inner[j]
                dx = xj - xi
                dy = yj - yi
                d = math.hypot(dx, dy)
                min_d = SQRT3 - 1e-7  # minimum center distance for non-overlap
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    max_overlap = max(max_overlap, overlap)
                    # Push apart along the center-to-center direction
                    nx = dx / d
                    ny = dy / d
                    push = overlap * 0.5
                    inner[i][0] -= push * nx
                    inner[i][1] -= push * ny
                    inner[j][0] += push * nx
                    inner[j][1] += push * ny
                elif d <= 1e-12:
                    # Coincident: push in arbitrary direction
                    overlap = min_d
                    max_overlap = max(max_overlap, overlap)
                    push = overlap * 0.5
                    inner[i][0] -= push
                    inner[j][0] += push
        if max_overlap < tol:
            break
    return inner


def pbd_resolve_container(inner, oc, oa, side, max_iters=30, tol=1e-8):
    """
    PBD: push hexagons inward if they protrude from the container.
    inner: list of [x, y, angle_deg] (mutable)
    oc: (cx, cy), oa: outer angle, side: outer side length
    """
    n = len(inner)
    ox, oy = oc
    for it in range(max_iters):
        if not _time_ok():
            break
        max_viol = 0.0
        base_angles = np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0]) * np.pi / 180.0
        normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
        ca = math.cos(math.radians(oa))
        sa = math.sin(math.radians(oa))
        R = np.array([[ca, -sa], [sa, ca]])
        
        for i in range(n):
            xi, yi, ai = inner[i]
            verts = hex_verts_array(xi, yi, 1.0, math.radians(ai))
            local = (verts - np.array([ox, oy])) @ R
            projs = local @ normals.T  # (6, 6)
            # For each face normal, the max projection must be <= side * SQRT3 / 2
            limit = side * SQRT3 / 2.0
            for k in range(6):
                p = projs[:, k]
                excess = p.max() - limit
                if excess > 0:
                    max_viol = max(max_viol, excess)
                    # Push the hexagon center inward along that normal
                    # The normal in world coords:
                    n_world = (R.T @ normals[k])
                    # Project vertices onto this normal to find which vertex is most protruding
                    # Simple: push center by excess along -n_world
                    inner[i][0] -= excess * n_world[0] * 0.5
                    inner[i][1] -= excess * n_world[1] * 0.5
        if max_viol < tol:
            break
    return inner


def pbd_iterate(inner, oc, oa, side, outer_iters=20, inner_iters=30):
    """
    Full PBD loop: alternate between resolving inner-inner collisions
    and inner-container violations.
    """
    for it in range(outer_iters):
        if not _time_ok():
            break
        # Resolve inner-inner
        pbd_resolve_collisions(inner, max_iters=inner_iters)
        # Resolve container
        pbd_resolve_container(inner, oc, oa, side, max_iters=10)
    return inner


# --- LP-based orientation optimization (from inspiration) ---

def orientation_lp(pts, beta):
    """Given fixed inner positions and outer orientation beta, find outer center + min S via LP."""
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    c = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            S = (res.x[2] + APOTHEM) / APOTHEM
            return S, (res.x[0], res.x[1])
    except Exception:
        pass
    return float('inf'), None


def best_orientation(pts):
    """Find best outer orientation via coarse sweep + refinement."""
    best_S = float('inf')
    best_beta = 0.0
    best_c = None
    for i in range(37):
        beta = i * (math.pi / 6) / 36
        S, c = orientation_lp(pts, beta)
        if S < best_S:
            best_S, best_beta, best_c = S, beta, c
    try:
        def obj(b):
            S, _ = orientation_lp(pts, b)
            return S
        res = minimize_scalar(obj, bounds=(max(0, best_beta - 0.03),
                                          min(math.pi / 6, best_beta + 0.03)),
                              method='bounded', options={'maxiter': 200, 'xatol': 1e-9})
        if res.fun < best_S:
            best_S = res.fun
            best_beta = res.x
            best_S, best_c = orientation_lp(pts, best_beta)
    except Exception:
        pass
    return best_S, best_beta, best_c


# --- Polish functions ---

def polish_local(p0, deadline, label=""):
    """Multi-stage local polish: smooth NM -> exact NM -> L-BFGS-B."""
    best_p = p0.copy()
    best_v = full_objective_exact(p0)

    # Stage 1: Nelder-Mead on smooth
    if time.time() < deadline - 8:
        try:
            r = minimize(full_objective_smooth, p0, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 4000, 'adaptive': True})
            if r.fun < best_v:
                best_v = r.fun
                best_p = r.x.copy()
        except Exception:
            pass

    # Stage 2: Nelder-Mead on exact
    if time.time() < deadline - 4:
        try:
            r = minimize(full_objective_exact, best_p, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 6000, 'adaptive': True})
            if r.fun < best_v:
                best_v = r.fun
                best_p = r.x.copy()
        except Exception:
            pass

    # Stage 3: L-BFGS-B
    if time.time() < deadline - 2:
        cx, cy = centroid([(best_p[i*3], best_p[i*3+1], best_p[i*3+2]) for i in range(N)])
        bounds = []
        for i in range(N):
            bounds.append((best_p[i*3] - 0.5, best_p[i*3] + 0.5))
            bounds.append((best_p[i*3+1] - 0.5, best_p[i*3+1] + 0.5))
            bounds.append((-180.0, 180.0))
        bounds.append((cx - 2.0, cx + 2.0))
        bounds.append((cy - 2.0, cy + 2.0))
        bounds.append((-30.0, 30.0))
        try:
            r = minimize(full_objective_exact, best_p, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000})
            if r.fun < best_v:
                best_v = r.fun
                best_p = r.x.copy()
        except Exception:
            pass

    return best_p, best_v


def rigid_optimize(axial, deadline):
    """Optimize treating all hexagons as rigid (shared rotation about centroid)."""
    s0 = 1.0 + 1e-9
    centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + 0.5 * q)) for q, r in axial]
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)
    base = [[x, y, 0.0] for x, y in centers]

    def rigid_obj(p):
        comp_a, dx, dy, oa = p[0], p[1], p[2], p[3]
        ca = math.cos(math.radians(comp_a))
        sa = math.sin(math.radians(comp_a))
        inner = []
        for (bx, by, _) in base:
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            inner.append((nx, ny, comp_a))
        return required_side_smooth(inner, (cx0 + dx, cy0 + dy), oa, beta=50.0)

    def rigid_obj_exact(p):
        comp_a, dx, dy, oa = p[0], p[1], p[2], p[3]
        ca = math.cos(math.radians(comp_a))
        sa = math.sin(math.radians(comp_a))
        inner = []
        for (bx, by, _) in base:
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            inner.append((nx, ny, comp_a))
        return required_side(inner, (cx0 + dx, cy0 + dy), oa)

    best_val = float('inf')
    best_p = None

    for comp_a in range(0, 60, 5):
        for oa in range(0, 60, 10):
            if time.time() > deadline - 10:
                return None, None, None, None
            p0 = np.array([float(comp_a), 0.0, 0.0, float(oa)])
            try:
                r1 = minimize(rigid_obj, p0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400})
                r2 = minimize(rigid_obj_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 700})
                if r2.fun < best_val:
                    best_val = r2.fun
                    best_p = r2.x.copy()
            except Exception:
                pass

    if best_p is None:
        return None, None, None, None

    comp_a, dx, dy, oa = best_p
    ca = math.cos(math.radians(comp_a))
    sa = math.sin(math.radians(comp_a))
    inner = []
    for (bx, by, _) in base:
        rx = bx - cx0
        ry = by - cy0
        nx = cx0 + ca * rx - sa * ry + dx
        ny = cy0 + sa * rx + ca * ry + dy
        inner.append([nx, ny, comp_a])
    oc = (cx0 + dx, cy0 + dy)
    return inner, oc, oa, best_val


def sa_optimize(p0, deadline, label="SA"):
    """Simulated annealing on full 36-DOF space."""
    current = p0.copy()
    current_val = full_objective_smooth(current)
    best_p = current.copy()
    best_val = current_val

    T = 0.5
    T_min = 0.0005
    cool = 0.9997
    step = 1.0
    np.random.seed(42 + int(time.time() * 1000) % 10000)

    while time.time() < deadline - 3:
        nb = current.copy()
        r = np.random.random()
        if r < 0.55:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, 0.15 * step)
        elif r < 0.72:
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, 6.0 * step)
        elif r < 0.82:
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, 0.08 * step)
        else:
            nb[N * 3 + 2] += np.random.normal(0, 2.0 * step)

        nb_val = full_objective_smooth(nb)
        delta = nb_val - current_val
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-12)):
            current = nb
            current_val = nb_val
            if current_val < best_val:
                best_val = current_val
                best_p = current.copy()
            step = min(2.0, step * 1.01)
        else:
            step = max(0.2, step * 0.985)

        T *= cool
        if T < T_min:
            break

    return best_p, best_val


def lattice_from_axial(axial, s=1.0 + 1e-9):
    """Generate hexagon positions from axial coordinates."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + 0.5 * q)) for q, r in axial]
    return [[x, y, 0.0] for x, y in centers]


def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    axial_seeds = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,-1),(1,1),(-2,1),(-1,-1)],
        [(-1,-1),(0,-1),(1,-1),(2,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(2,1)],
        [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)],
        [(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)],
        [(0,0),(1,0),(2,0),(3,0),(-1,1),(0,1),(1,1),(2,1),(-1,-1),(0,-1),(1,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(0,2),(-2,1),(-2,-1)],
    ]

    best_side = float('inf')
    best_inner = None
    best_oc = None
    best_oa = 0.0
    best_params = None

    # Phase 1: Rigid composite optimization per seed
    for si, axial in enumerate(axial_seeds):
        if time.time() > deadline - 85:
            break
        inner, oc, oa, side_val = rigid_optimize(axial, deadline)
        if inner is not None and check_disjoint(inner, 1e-5) and side_val < best_side:
            best_side = side_val
            best_inner = [list(x) for x in inner]
            best_oc = oc
            best_oa = oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(inner):
                p0[i*3] = x
                p0[i*3+1] = y
                p0[i*3+2] = a
            p0[N*3] = oc[0]
            p0[N*3+1] = oc[1]
            p0[N*3+2] = oa
            best_params = p0.copy()

    if best_inner is None:
        best_inner = lattice_from_axial(axial_seeds[0])
        cx, cy = centroid(best_inner)
        best_oc = (cx, cy)
        best_oa = 0.0
        best_side = required_side(best_inner, best_oc, best_oa)
        best_params = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            best_params[i*3] = x
            best_params[i*3+1] = y
            best_params[i*3+2] = a
        best_params[N*3] = cx
        best_params[N*3+1] = cy
        best_params[N*3+2] = 0.0

    # Phase 2: PBD constraint projection to fix any residual overlaps
    if time.time() < deadline - 30:
        pbd_inner = [list(x) for x in best_inner]
        pbd_resolve_collisions(pbd_inner, max_iters=100, tol=1e-9)
        if check_disjoint(pbd_inner, 1e-5):
            # Re-optimize container after PBD
            pts = [(p[0], p[1]) for p in pbd_inner]
            S_lp, beta, c_lp = best_orientation(pts)
            if c_lp is not None and S_lp < best_side:
                best_side = S_lp
                best_inner = pbd_inner
                best_oc = c_lp
                best_oa = math.degrees(beta)
                p0 = np.zeros(N * 3 + 3)
                for i, (x, y, a) in enumerate(pbd_inner):
                    p0[i*3] = x
                    p0[i*3+1] = y
                    p0[i*3+2] = a
                p0[N*3] = c_lp[0]
                p0[N*3+1] = c_lp[1]
                p0[N*3+2] = math.degrees(beta)
                best_params = p0.copy()

    # Phase 3: SA optimization
    if time.time() < deadline - 20 and best_params is not None:
        sa_p, sa_val = sa_optimize(best_params, deadline - 8)
        sa_inner = []
        for i in range(N):
            sa_inner.append([float(sa_p[i*3]), float(sa_p[i*3+1]), float(sa_p[i*3+2])])
        sa_oc = (float(sa_p[N*3]), float(sa_p[N*3+1]))
        sa_oa = float(sa_p[N*3+2])
        if check_disjoint(sa_inner, 1e-5):
            sa_side = required_side(sa_inner, sa_oc, sa_oa)
            if sa_side < best_side:
                best_side = sa_side
                best_inner = sa_inner
                best_oc = sa_oc
                best_oa = sa_oa
                best_params = sa_p.copy()

    # Phase 4: PBD polish on best solution
    if time.time() < deadline - 10:
        pbd_inner2 = [list(x) for x in best_inner]
        pbd_resolve_collisions(pbd_inner2, max_iters=200, tol=1e-10)
        if check_disjoint(pbd_inner2, 1e-5):
            pts2 = [(p[0], p[1]) for p in pbd_inner2]
            S_lp2, beta2, c_lp2 = best_orientation(pts2)
            if c_lp2 is not None and S_lp2 < best_side:
                best_side = S_lp2
                best_inner = pbd_inner2
                best_oc = c_lp2
                best_oa = math.degrees(beta2)
                p0 = np.zeros(N * 3 + 3)
                for i, (x, y, a) in enumerate(pbd_inner2):
                    p0[i*3] = x
                    p0[i*3+1] = y
                    p0[i*3+2] = a
                p0[N*3] = c_lp2[0]
                p0[N*3+1] = c_lp2[1]
                p0[N*3+2] = math.degrees(beta2)
                best_params = p0.copy()

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5 and best_params is not None:
        polished_p, polished_val = polish_local(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished_p[i*3]), float(polished_p[i*3+1]), float(polished_p[i*3+2])])
        polish_oc = (float(polished_p[N*3]), float(polished_p[N*3+1]))
        polish_oa = float(polished_p[N*3+2])
        if check_disjoint(polish_inner, 1e-5):
            polish_side = required_side(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_side = polish_side
                best_inner = polish_inner
                best_oc = polish_oc
                best_oa = polish_oa

    # Final validation
    if not check_disjoint(best_inner, 1e-5):
        best_inner = lattice_from_axial(axial_seeds[0])
        cx, cy = centroid(best_inner)
        best_oc = (cx, cy)
        best_oa = 0.0
        best_side = required_side(best_inner, best_oc, best_oa)

    return best_inner, best_oc, best_side, best_oa
