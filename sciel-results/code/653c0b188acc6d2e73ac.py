"""
Improved hexagon packing: 11 unit hexagons in minimal outer hexagon.
Key changes: better lattice seeds, L-BFGS-B with penalty, tighter final polish,
vectorized SAT, and more diverse global search via DE-like restarts.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution

N = 11
SQRT3 = math.sqrt(3)
TWO_PI = 2 * math.pi

# EVOLVE-BLOCK-START

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
    # Outer hexagon support: 6 face normals at 30,90,150,210,270,330
    base_angles = np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0]) * np.pi / 180.0
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])  # (6,2)
    
    # Rotation of outer
    ca = math.cos(math.radians(oa))
    sa = math.sin(math.radians(oa))
    R = np.array([[ca, sa], [-sa, ca]])  # rotate world->outer-local
    
    max_proj = 0.0
    for (ix, iy, iang) in inner:
        verts = hex_verts_array(ix, iy, 1.0, math.radians(iang))  # (6,2)
        local = (verts - np.array([ox, oy])) @ R.T  # (6,2)
        projs = local @ normals.T  # (6,6)
        m = projs.max()
        if m > max_proj:
            max_proj = m
    return 2.0 * max_proj / SQRT3

def required_side_smooth(inner, oc, oa, beta=60.0):
    """Smooth surrogate via log-sum-exp."""
    ox, oy = oc
    base_angles = np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0]) * np.pi / 180.0
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    ca = math.cos(math.radians(oa))
    sa = math.sin(math.radians(oa))
    R = np.array([[ca, sa], [-sa, ca]])
    
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_verts_array(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
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
    """Separating axis theorem for two convex polygons (each list of (x,y))."""
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

def lattice_from_axial(axial, s=1.0 + 1e-9):
    """Generate hexagon positions from axial coordinates with spacing s."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + 0.5 * q)) for q, r in axial]
    return [[x, y, 0.0] for x, y in centers]

def centroid(inner):
    n = len(inner)
    cx = sum(p[0] for p in inner) / n
    cy = sum(p[1] for p in inner) / n
    return cx, cy

def full_objective_smooth(p, beta=50.0):
    """Objective for optimization: smooth side + penalty.
    p: 3N + 3 = 36 params: [x0,y0,a0, x1,y1,a1, ..., ox, oy, oa]"""
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

def polish_local(p0, deadline, label=""):
    """Multi-stage local polish: smooth NM -> exact NM -> exact L-BFGS-B."""
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
    
    # Stage 3: L-BFGS-B with small bounds (centered on current)
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
    """Optimize treating all hexagons as rigid (shared rotation about centroid).
    5 DOF: comp_angle, dx, dy, oa, plus we fix ox,oy to centroid."""
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
        # Outer centered on composite centroid
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
    
    # Grid of starting angles
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

def semi_rigid_optimize(axial, deadline):
    """Semi-rigid: fixed positions, individual angles + outer. 14 DOF."""
    s0 = 1.0 + 1e-9
    centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + 0.5 * q)) for q, r in axial]
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)
    base = [[x, y] for x, y in centers]
    
    def semi_obj(p):
        angles = p[:N]
        ox, oy, oa = p[N], p[N+1], p[N+2]
        inner = [(base[i][0], base[i][1], angles[i]) for i in range(N)]
        return required_side_smooth(inner, (ox, oy), oa, beta=40.0)
    
    def semi_obj_exact(p):
        angles = p[:N]
        ox, oy, oa = p[N], p[N+1], p[N+2]
        inner = [(base[i][0], base[i][1], angles[i]) for i in range(N)]
        return required_side(inner, (ox, oy), oa)
    
    best_val = float('inf')
    best_p = None
    
    for oa0 in range(0, 60, 15):
        if time.time() > deadline - 8:
            return None, None, None, None
        np.random.seed(oa0 * 7 + 13)
        p0 = np.zeros(N + 3)
        p0[:N] = np.random.uniform(-20, 20, N)
        p0[N] = cx0
        p0[N+1] = cy0
        p0[N+2] = float(oa0)
        try:
            r1 = minimize(semi_obj, p0, method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 600})
            r2 = minimize(semi_obj_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 600, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 900})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    
    if best_p is None:
        return None, None, None, None
    
    angles = best_p[:N]
    ox, oy, oa = best_p[N], best_p[N+1], best_p[N+2]
    inner = [[base[i][0], base[i][1], float(angles[i])] for i in range(N)]
    return inner, (float(ox), float(oy)), float(oa), best_val

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
    
    it = 0
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
        it += 1
        if T < T_min:
            break
    
    return best_p, best_val

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin from 150s limit
    
    # Multiple lattice seeds
    axial_seeds = [
        # Seed 1: center + ring + 4 extra
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,-1),(1,1),(-2,1),(-1,-1)],
        # Seed 2: 4-3-4 rows
        [(-1,-1),(0,-1),(1,-1),(2,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(2,1)],
        # Seed 3: 3-5-3 rows
        [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)],
        # Seed 4: 5-3-3
        [(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)],
        # Seed 5: elongated
        [(0,0),(1,0),(2,0),(3,0),(-1,1),(0,1),(1,1),(2,1),(-1,-1),(0,-1),(1,-1)],
        # Seed 6: compact
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
            best_inner = inner
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
    
    # Phase 2: Semi-rigid for top seeds
    for si in range(min(3, len(axial_seeds))):
        if time.time() > deadline - 60:
            break
        inner, oc, oa, side_val = semi_rigid_optimize(axial_seeds[si], deadline)
        if inner is not None and check_disjoint(inner, 1e-5) and side_val < best_side:
            best_side = side_val
            best_inner = inner
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
        # Fallback
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
    
    # Phase 3: Full SA optimization from best
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
    
    # Phase 4: Final high-precision polish
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

# EVOLVE-BLOCK-END
