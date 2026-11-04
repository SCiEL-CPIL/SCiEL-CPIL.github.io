"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 116.0  # safety margin below 150s kill

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """True if two convex polygons overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _outer_hex_side(centers_angles, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

# ---- Smooth SDF machinery (from inspiration) ----
def _sdf_hex_point(px, py, angle_deg, side=1.0):
    """Signed distance from point (px,py) to hexagon centered at origin with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    lx = ca * px + sa * py
    ly = -sa * px + ca * py
    apothem = side * math.sqrt(3) / 2.0
    inside = True
    max_proj = 0.0
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx, ny = math.cos(theta), math.sin(theta)
        proj = nx * lx + ny * ly
        if proj > max_proj:
            max_proj = proj
        if proj > apothem + 1e-12:
            inside = False
    if inside:
        return -(apothem - max_proj)
    min_d = float('inf')
    for k in range(6):
        t1 = math.radians(60 * k)
        t2 = math.radians(60 * (k + 1))
        v1 = (side * math.cos(t1), side * math.sin(t1))
        v2 = (side * math.cos(t2), side * math.sin(t2))
        dx, dy = v2[0] - v1[0], v2[1] - v1[1]
        len2 = dx * dx + dy * dy
        if len2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((lx - v1[0]) * dx + (ly - v1[1]) * dy) / len2))
        qx, qy = v1[0] + t * dx, v1[1] + t * dy
        d = math.hypot(lx - qx, ly - qy)
        if d < min_d:
            min_d = d
    return min_d

def _sdf_hex_hex(cx1, cy1, a1, cx2, cy2, a2, side=1.0):
    """Approximate min signed distance between two hexagons (positive = separated)."""
    verts1 = _hex_vertices(cx1, cy1, a1, side)
    verts2 = _hex_vertices(cx2, cy2, a2, side)
    min_d = float('inf')
    for (vx, vy) in verts1:
        d = _sdf_hex_point(vx - cx2, vy - cy2, a2, side)
        if d < min_d:
            min_d = d
    for (vx, vy) in verts2:
        d = _sdf_hex_point(vx - cx1, vy - cy1, a1, side)
        if d < min_d:
            min_d = d
    return min_d

def _collision_penalty(cfg, sigma=0.08):
    m = len(cfg)
    total = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            d = _sdf_hex_hex(cfg[i][0], cfg[i][1], cfg[i][2],
                             cfg[j][0], cfg[j][1], cfg[j][2])
            if d < 3.0 * sigma:
                # smooth barrier: large when overlapping (d<0), ->0 when d>>sigma
                total += math.exp(-d / sigma) - 1.0
    return total

def _is_valid(centers_angles):
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _flatten(cfg, center, angle):
    v = []
    for p in cfg:
        v.extend([p[0], p[1], p[2]])
    v.extend([center[0], center[1], angle])
    return v

def _unflatten(v):
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = [v[3*N], v[3*N+1]]
    angle = v[3*N+2]
    return cfg, center, angle

def _smooth_objective(v, sigma=0.08):
    """Differentiable surrogate: outer side + smooth collision penalty."""
    cfg, center, angle = _unflatten(v)
    s = _outer_hex_side(cfg, tuple(center), angle)
    pen = _collision_penalty(cfg, sigma)
    return s + 100.0 * pen

def _hex_lattice(rows, cols):
    pts = []
    row_height = math.sqrt(3)
    col_width = 2.0
    for r in range(rows):
        for c in range(cols):
            x = c * col_width + (col_width / 2 if r % 2 else 0.0)
            y = r * row_height
            pts.append([x - (cols-1)*col_width/2.0, y - (rows-1)*row_height/2.0, 0.0])
    return pts

def _ring(R):
    pts = []
    for i in range(N):
        theta = 2 * math.pi * i / N
        pts.append([R * math.cos(theta), R * math.sin(theta), math.degrees(theta)])
    return pts

def _cluster():
    pts = [[0.0, 0.0, 0.0]]
    R1 = math.sqrt(3)
    for i in range(6):
        theta = 2 * math.pi * i / 6
        pts.append([R1 * math.cos(theta), R1 * math.sin(theta), 0.0])
    R2 = 2.0 * math.sqrt(3)
    for i in range(5):
        theta = 2 * math.pi * (i + 0.5) / 5
        pts.append([R2 * math.cos(theta), R2 * math.sin(theta), 0.0])
    return pts

def _full_score(cfg, center, angle):
    if not _is_valid(cfg):
        return None
    return _outer_hex_side(cfg, tuple(center), angle)

def _lbfgs_polish(cfg0, center0, angle0, time_budget, sigma=0.08):
    """L-BFGS-B on the smooth surrogate; returns best valid (cfg, center, angle, s)."""
    v0 = _flatten(cfg0, center0, angle0)
    t0 = time.time()
    best_v, best_s = v0[:], None
    # Try a few sigma values (anneal the barrier)
    for sig in (sigma, max(0.02, sigma*0.4)):
        if time.time() - t0 > time_budget:
            break
        def obj(v):
            return _smooth_objective(v, sig)
        try:
            res = minimize(obj, v0, method='L-BFGS-B',
                           options={'maxiter': 400, 'maxfun': 4000,
                                    'gtol': 1e-6, 'maxls': 20})
        except Exception:
            break
        for cand in (res.x, v0):
            cc, cctr, ca = _unflatten(cand)
            s = _full_score(cc, cctr, ca)
            if s is not None:
                if best_s is None or s < best_s:
                    best_s = s
                    best_v = cand[:]
        # continue from the result for the next sigma
        if res.success or res.fun < obj(v0):
            v0 = res.x
    if best_s is None:
        return None
    return _unflatten(best_v) + [best_s]

def _nelder_mead_polish(cfg0, center0, angle0, time_budget):
    def obj(v):
        cfg, center, angle = _unflatten(v)
        s = _full_score(cfg, center, angle)
        return 1e6 if s is None else s
    v0 = _flatten(cfg0, center0, angle0)
    n = len(v0)
    simplex = [v0[:]]
    step = 0.05
    for i in range(n):
        vi = v0[:]
        vi[i] += step
        simplex.append(vi)
    alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5
    best_v, best_f = None, float('inf')
    t0 = time.time()
    iters = 0
    while iters < 1500 and time.time() - t0 < time_budget:
        simplex.sort(key=obj)
        f_best = obj(simplex[0])
        if f_best < best_f:
            best_f = f_best
            best_v = simplex[0][:]
        f_max = obj(simplex[-1])
        if abs(f_max - f_best) < 1e-9:
            break
        centroid = [sum(v[i] for v in simplex[:-1]) / n for i in range(n)]
        xr = [centroid[i] + alpha_nm*(centroid[i]-simplex[-1][i]) for i in range(n)]
        fr = obj(xr)
        if fr < f_best:
            xe = [centroid[i] + gamma_nm*(centroid[i]-simplex[-1][i]) for i in range(n)]
            fe = obj(xe)
            simplex[-1] = xe if fe < fr else xr
        elif fr < obj(simplex[-2]):
            simplex[-1] = xr
        else:
            if fr < obj(simplex[-1]):
                xc = [centroid[i] + rho_nm*(simplex[-1][i]-centroid[i]) for i in range(n)]
            else:
                xc = [centroid[i] + rho_nm*(centroid[i]-simplex[-1][i]) for i in range(n)]
            fc = obj(xc)
            if fc < min(fr, obj(simplex[-1])):
                simplex[-1] = xc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = [simplex[0][j] + sigma_nm*(simplex[i][j]-simplex[0][j]) for j in range(n)]
        iters += 1
    if best_v is None or best_f >= 1e5:
        return None
    cfg, center, angle = _unflatten(best_v)
    return cfg, center, angle, best_f

def _grid_search_outer(cfg_fixed, cx0, cy0, ang0, time_budget):
    local_s = float('inf')
    local_cx, local_cy, local_ang = cx0, cy0, ang0
    t0 = time.time()
    for (cxr, cyr, ar, n) in ((1.0, 1.0, math.pi/6.0, 10), (0.3, 0.3, math.pi/24.0, 14)):
        for i in range(n):
            cx = local_cx - cxr + 2*cxr*i/(n-1)
            for j in range(n):
                cy = local_cy - cyr + 2*cyr*j/(n-1)
                for k in range(n):
                    ang = local_ang - ar + 2*ar*k/(n-1)
                    if time.time() - t0 > time_budget:
                        return local_s, local_cx, local_cy, local_ang
                    s = _outer_hex_side(cfg_fixed, (cx, cy), ang)
                    if s < local_s:
                        local_s, local_cx, local_cy, local_ang = s, cx, cy, ang
    return local_s, local_cx, local_cy, local_ang

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0

    inits = [_hex_lattice(3, 4), _ring(2.5), _cluster(), _hex_lattice(4, 3)]
    for init in inits:
        if len(init) < N:
            continue
        init = init[:N]
        s = _full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0

    if best_cfg is None:
        best_cfg = [[random.uniform(-3, 3), random.uniform(-3, 3), 0.0] for _ in range(N)]
        best_score = 50.0

    # ---- Simulated annealing ----
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    T = 0.8
    T_end = 0.003
    alpha = 0.9997

    def _restore(r, old):
        if r < 0.50:
            cfg[old['idx']][0], cfg[old['idx']][1] = old['x'], old['y']
        elif r < 0.70:
            cfg[old['idx']][2] = old['ang']
        elif r < 0.80:
            center[0], center[1] = old['cx'], old['cy']
        elif r < 0.90:
            angle = old['angle']
        else:
            cfg[old['i1']][0], cfg[old['i1']][1] = old['x1'], old['y1']
            cfg[old['i2']][0], cfg[old['i2']][1] = old['x2'], old['y2']

    while time.time() - t0 < TIME_LIMIT - 45.0:
        r = random.random()
        old = {}
        if r < 0.50:
            idx = random.randrange(N)
            old['idx'], old['x'], old['y'] = idx, cfg[idx][0], cfg[idx][1]
            step = 0.2 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.70:
            idx = random.randrange(N)
            old['idx'], old['ang'] = idx, cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
        elif r < 0.80:
            old['cx'], old['cy'] = center[0], center[1]
            step = 0.1 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.90:
            old['angle'] = angle
            angle += random.gauss(0, 2.0 + 2.0 * T)
        else:
            i1, i2 = random.sample(range(N), 2)
            old['i1'], old['i2'] = i1, i2
            old['x1'], old['y1'] = cfg[i1][0], cfg[i1][1]
            old['x2'], old['y2'] = cfg[i2][0], cfg[i2][1]
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[i1][0] += random.gauss(0, step); cfg[i1][1] += random.gauss(0, step)
            cfg[i2][0] += random.gauss(0, step); cfg[i2][1] += random.gauss(0, step)

        new_score = _full_score(cfg, tuple(center), angle)
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score)/T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                _restore(r, old)
        else:
            _restore(r, old)
        T = max(T_end, T * alpha)

    # ---- Refinement passes, each guarded by time, each may improve best ----
    def _try_update(res):
        nonlocal best_cfg, best_score, best_center, best_angle
        if res is None:
            return
        rcfg, rcenter, rangle, rs = res
        if rs is not None and rs < best_score:
            best_cfg = [row[:] for row in rcfg]
            best_center = rcenter[:]
            best_angle = rangle
            best_score = rs

    # L-BFGS-B smooth surrogate polish
    if time.time() - t0 < TIME_LIMIT - 20.0:
        _try_update(_lbfgs_polish(best_cfg, best_center, best_angle,
                                  min(20.0, TIME_LIMIT - (time.time()-t0) - 20.0)))

    # Nelder-Mead polish (now actually used)
    if time.time() - t0 < TIME_LIMIT - 10.0:
        _try_update(_nelder_mead_polish(best_cfg, best_center, best_angle,
                                        min(12.0, TIME_LIMIT - (time.time()-t0) - 10.0)))

    # One more L-BFGS-B pass from the NM result (different basin)
    if time.time() - t0 < TIME_LIMIT - 8.0:
        _try_update(_lbfgs_polish(best_cfg, best_center, best_angle,
                                  min(8.0, TIME_LIMIT - (time.time()-t0) - 8.0), sigma=0.03))

    # Final outer-hex grid search (inner fixed, already valid)
    if time.time() - t0 < TIME_LIMIT - 4.0:
        gs_s, gs_cx, gs_cy, gs_ang = _grid_search_outer(
            best_cfg, best_center[0], best_center[1], best_angle,
            TIME_LIMIT - (time.time()-t0) - 4.0)
        if gs_s < best_score:
            best_score = gs_s
            best_center = [gs_cx, gs_cy]
            best_angle = gs_ang

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
