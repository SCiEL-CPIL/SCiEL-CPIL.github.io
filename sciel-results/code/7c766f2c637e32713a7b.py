"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random

N = 12
TIME_LIMIT = 112.0  # safety margin below 120s (evaluator kills at 150s)

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
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
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
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
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

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def construct():
    t0 = time.time()
    random.seed(42)

    def make_hex_lattice():
        """3x4 hexagonal lattice with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts

    def make_ring():
        """12 hexagons in a ring, oriented outward."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts

    def make_cluster():
        """Tight cluster: central hex + ring of 6 + outer 5."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = 2.0 * math.sqrt(3) / 2.0
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2.0 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)

    # ---- Try multiple initial configurations, keep the best valid one ----
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0

    inits = [make_hex_lattice(), make_ring(), make_cluster()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0

    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0

    # ---- Simulated annealing phase ----
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score

    T = 0.8
    T_end = 0.003
    alpha = 0.9997

    while time.time() - t0 < TIME_LIMIT - 25.0:  # reserve ~25s for polish phases
        r = random.random()

        if r < 0.50:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.2 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.70:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
        elif r < 0.80:
            old_cx, old_cy = center[0], center[1]
            step = 0.1 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.90:
            old_angle = angle
            angle += random.gauss(0, 2.0 + 2.0 * T)
        else:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)

        new_score = full_score(cfg, tuple(center), angle)

        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                if r < 0.50:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.70:
                    cfg[idx][2] = old_ang
                elif r < 0.80:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.90:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        else:
            if r < 0.50:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.70:
                cfg[idx][2] = old_ang
            elif r < 0.80:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.90:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2

        T = max(T_end, T * alpha)

    # ---- Nelder-Mead polish on the full 39-parameter vector ----
    def nelder_mead_polish(cfg_init, center_init, angle_init, time_budget):
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return v

        def unflatten(v):
            cfg = []
            for i in range(N):
                cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
            center = [v[3*N], v[3*N+1]]
            angle = v[3*N+2]
            return cfg, center, angle

        def objective(v):
            cfg, center, angle = unflatten(v)
            s = full_score(cfg, tuple(center), angle)
            if s is None:
                return 1e6
            return s

        v0 = flatten(cfg_init, center_init, angle_init)
        n_params = len(v0)
        simplex = [v0[:]]
        step = 0.05
        for i in range(n_params):
            vi = v0[:]
            vi[i] += step
            simplex.append(vi)

        alpha_nm = 1.0
        gamma_nm = 2.0
        rho_nm = 0.5
        sigma_nm = 0.5

        best_f = min(objective(v) for v in simplex)
        best_v = min(simplex, key=objective)

        max_iters = 2000
        iter_count = 0
        while iter_count < max_iters and time.time() - t0 < time_budget:
            simplex.sort(key=lambda v: objective(v))
            f_best = objective(simplex[0])

            if f_best < best_f:
                best_f = f_best
                best_v = simplex[0][:]

            f_max = objective(simplex[-1])
            if abs(f_max - f_best) < 1e-8:
                break

            centroid = [sum(v[i] for v in simplex[:-1]) / n_params for i in range(n_params)]

            xr = [centroid[i] + alpha_nm * (centroid[i] - simplex[-1][i]) for i in range(n_params)]
            fr = objective(xr)

            if fr < f_best:
                xe = [centroid[i] + gamma_nm * (centroid[i] - simplex[-1][i]) for i in range(n_params)]
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
            elif fr < objective(simplex[-2]):
                simplex[-1] = xr
            else:
                if fr < objective(simplex[-1]):
                    xc = [centroid[i] + rho_nm * (simplex[-1][i] - centroid[i]) for i in range(n_params)]
                else:
                    xc = [centroid[i] + rho_nm * (centroid[i] - simplex[-1][i]) for i in range(n_params)]
                fc = objective(xc)
                if fc < min(fr, objective(simplex[-1])):
                    simplex[-1] = xc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = [simplex[0][j] + sigma_nm * (simplex[i][j] - simplex[0][j]) for j in range(n_params)]

            iter_count += 1

        return best_v, best_f

    nm_budget = min(TIME_LIMIT - 10.0, time.time() - t0 + 15.0)
    best_v, nm_f = nelder_mead_polish(best_cfg, best_center, best_angle, nm_budget)
    if nm_f < 1e5:
        nm_cfg, nm_center, nm_angle = _unflatten_best(best_v)
        if nm_f < best_score:
            best_score = nm_f
            best_cfg = nm_cfg
            best_center = nm_center
            best_angle = nm_angle

    # ---- Final grid-search polish over outer center and rotation ----
    # The inner configuration is fixed (already valid). We only vary the outer
    # hexagon's center (cx, cy) and rotation angle to minimize the bounding side.
    def grid_search_outer(cfg_fixed, cx0, cy0, ang0, time_budget):
        """Fine grid search over outer center and rotation, keeping inner fixed."""
        local_best_s = float('inf')
        local_best_cx, local_best_cy, local_best_ang = cx0, cy0, ang0

        # Determine search range around current best
        cx_range = 1.0
        cy_range = 1.0
        ang_range = math.pi / 6.0  # 30 degrees (hex symmetry)

        # Coarse pass
        n_coarse = 12
        for i in range(n_coarse):
            cx = cx0 - cx_range + 2 * cx_range * i / (n_coarse - 1)
            for j in range(n_coarse):
                cy = cy0 - cy_range + 2 * cy_range * j / (n_coarse - 1)
                for k in range(n_coarse):
                    ang = ang0 - ang_range + 2 * ang_range * k / (n_coarse - 1)
                    if time.time() - t0 > time_budget:
                        break
                    s = _outer_hex_side(cfg_fixed, (cx, cy), ang)
                    if s < local_best_s:
                        local_best_s = s
                        local_best_cx, local_best_cy, local_best_ang = cx, cy, ang
                if time.time() - t0 > time_budget:
                    break
            if time.time() - t0 > time_budget:
                break

        # Fine pass around the coarse best
        cx_range = 0.3
        cy_range = 0.3
        ang_range = math.pi / 24.0
        n_fine = 15
        for i in range(n_fine):
            cx = local_best_cx - cx_range + 2 * cx_range * i / (n_fine - 1)
            for j in range(n_fine):
                cy = local_best_cy - cy_range + 2 * cy_range * j / (n_fine - 1)
                for k in range(n_fine):
                    ang = local_best_ang - ang_range + 2 * ang_range * k / (n_fine - 1)
                    if time.time() - t0 > time_budget:
                        break
                    s = _outer_hex_side(cfg_fixed, (cx, cy), ang)
                    if s < local_best_s:
                        local_best_s = s
                        local_best_cx, local_best_cy, local_best_ang = cx, cy, ang
                if time.time() - t0 > time_budget:
                    break
            if time.time() - t0 > time_budget:
                break

        return local_best_s, local_best_cx, local_best_cy, local_best_ang

    gs_budget = TIME_LIMIT - 5.0
    if time.time() - t0 < gs_budget:
        gs_s, gs_cx, gs_cy, gs_ang = grid_search_outer(
            best_cfg, best_center[0], best_center[1], best_angle, gs_budget
        )
        if gs_s < best_score:
            best_score = gs_s
            best_center = [gs_cx, gs_cy]
            best_angle = gs_ang

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

def _unflatten_best(v):
    """Unflatten a 39-param vector into (cfg, center, angle)."""
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = [v[3*N], v[3*N+1]]
    angle = v[3*N+2]
    return cfg, center, angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
