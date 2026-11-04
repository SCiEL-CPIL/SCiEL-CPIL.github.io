import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0
APOTHEM = math.sqrt(3) / 2.0
# Circumscribed radius of unit hexagon
CIRC_RADIUS = 1.0


def _hex_vertices(cx, cy, angle_deg, side=1.0):
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


def _sat_overlap(vertsA, vertsB):
    """Return True if two convex polygons overlap (interiors intersect)."""
    # Fast bounding circle pre-check
    cx_a = sum(v[0] for v in vertsA) / 6
    cy_a = sum(v[1] for v in vertsA) / 6
    cx_b = sum(v[0] for v in vertsB) / 6
    cy_b = sum(v[1] for v in vertsB) / 6
    dx, dy = cx_b - cx_a, cy_b - cy_a
    if dx * dx + dy * dy > 4.0:
        return False

    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True


def _is_valid(config):
    """config: list of [x, y, angle_deg]. Return True if all pairs disjoint."""
    polys = [_hex_vertices(c[0], c[1], c[2]) for c in config]
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(polys[i], polys[j]):
                return False
    return True


def _outer_hex_side(config, center, angle_deg):
    a = math.radians(angle_deg)
    cx, cy = center
    # Outer hexagon side normals (apothem direction)
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))

    max_need = 0.0
    for (px, py, pa) in config:
        verts = _hex_vertices(px, py, pa)
        for (vx, vy) in verts:
            rx = vx - cx
            ry = vy - cy
            for (nx, ny) in normals:
                val = (nx * rx + ny * ry) / APOTHEM
                if val > max_need:
                    max_need = val
    return max_need


def _make_hex_tiling_initial():
    """Compact cluster of 12 hexagons on triangular lattice."""
    dx = math.sqrt(3)
    dy = 1.5
    positions = [
        (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
        (1, -1), (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    pts = []
    for q, r in positions:
        x = dx * (q + r / 2.0)
        y = dy * r
        pts.append([x, y, 0.0])
    return pts


def _make_finite_ring_initial():
    """Center + ring of 6 + outer partial ring."""
    pts = [[0.0, 0.0, 0.0]]
    r1 = math.sqrt(3)
    for i in range(6):
        ang = math.radians(60 * i)
        pts.append([r1 * math.cos(ang), r1 * math.sin(ang), 0.0])
    r2 = r1 + math.sqrt(3)
    for i in range(5):
        ang = math.radians(30 + 72 * i)
        pts.append([r2 * math.cos(ang), r2 * math.sin(ang), 0.0])
    return pts


def _make_random_initial():
    pts = []
    for _ in range(N):
        pts.append([random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 60)])
    return pts


def construct():
    t0 = time.time()
    random.seed(123)

    def time_left():
        return TIME_LIMIT - (time.time() - t0)

    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # --- Phase 1: Evaluate diverse initial configurations ---
    inits = [
        _make_hex_tiling_initial(),
        _make_finite_ring_initial(),
        _make_random_initial(),
        _make_random_initial(),
        _make_random_initial(),
        _make_hex_tiling_initial(),  # same but we'll perturb
    ]
    for _ in range(2):
        inits.append(_make_random_initial())

    for cfg in inits:
        if time_left() < 10:
            break
        if len(cfg) != N:
            continue
        # Small perturbation to break symmetry
        for p in cfg:
            p[0] += random.gauss(0, 0.05)
            p[1] += random.gauss(0, 0.05)

        if _is_valid(cfg):
            side = _outer_hex_side(cfg, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = [row[:] for row in cfg]

    # Fallback if no valid initial
    if best_cfg is None:
        cfg = _make_hex_tiling_initial()
        # Scale up to ensure validity
        for p in cfg:
            p[0] *= 1.1
            p[1] *= 1.1
        best_cfg = cfg
        best_side = _outer_hex_side(cfg, (0.0, 0.0), 0.0)

    # --- Phase 2: Smooth SDF-based objective with L-BFGS-B ---
    def _sdf_hex_point(px, py, cx, cy, angle_deg):
        """Approximate SDF from point to hexagon. Negative = inside."""
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rx = ca * (px - cx) + sa * (py - cy)
        ry = -sa * (px - cx) + ca * (py - cy)
        ax, ay = abs(rx), abs(ry)
        # Hexagon SDF approximation: use min of 6 half-space distances
        d = 0.0
        for k in range(6):
            theta = math.radians(60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            # Distance to edge line (signed, outward normal)
            dist = nx * rx + ny * ry - APOTHEM
            if dist > d:
                d = dist
        return d

    def _smooth_objective(params):
        """params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, cx, cy, ca] -> 38 values"""
        cfg = []
        for i in range(N):
            x, y, a = params[3 * i], params[3 * i + 1], params[3 * i + 2]
            cfg.append([x, y, a])
        cx, cy, ca = params[3 * N], params[3 * N + 1], params[3 * N + 2]

        # Outer side length
        side = _outer_hex_side(cfg, (cx, cy), ca)

        # Collision penalty: smooth approximation using center distances
        # Minimum center distance for non-overlap depends on relative orientation.
        # For same orientation: sqrt(3). We use a conservative lower bound of 1.7.
        penalty = 0.0
        min_dist = 1.7  # Conservative: less than sqrt(3)=1.732
        for i in range(N):
            xi, yi = cfg[i][0], cfg[i][1]
            for j in range(i + 1, N):
                xj, yj = cfg[j][0], cfg[j][1]
                d = math.hypot(xi - xj, yi - yj)
                if d < min_dist:
                    # Smooth penalty
                    penalty += (min_dist - d) ** 2

        # Weight: large penalty to discourage overlap
        return side + 200.0 * penalty

    def _refine_lbfgs(cfg_init, max_time):
        """Run L-BFGS-B optimization starting from cfg_init."""
        nonlocal best_side, best_cfg, best_center, best_angle

        # Initial params
        params = np.array([v for p in cfg_init for v in p] + [0.0, 0.0, 0.0])

        def obj(p):
            return _smooth_objective(p)

        # Run L-BFGS-B
        result = minimize(
            obj, params,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-10}
        )
        if result.success or result.fun < _smooth_objective(params):
            p_opt = result.x
            cfg_opt = [[p_opt[3 * i], p_opt[3 * i + 1], p_opt[3 * i + 2]] for i in range(N)]
            cx, cy, ca = p_opt[3 * N], p_opt[3 * N + 1], p_opt[3 * N + 2]

            # Check validity
            if _is_valid(cfg_opt):
                side = _outer_hex_side(cfg_opt, (cx, cy), ca)
                if side < best_side:
                    best_side = side
                    best_cfg = [row[:] for row in cfg_opt]
                    best_center = (cx, cy)
                    best_angle = ca
            else:
                # Try with original center/angle
                if _is_valid(cfg_opt):
                    side = _outer_hex_side(cfg_opt, (0.0, 0.0), 0.0)
                    if side < best_side:
                        best_side = side
                        best_cfg = [row[:] for row in cfg_opt]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0

    # Run L-BFGS from multiple starting points
    start_configs = [best_cfg]
    for cfg in inits[:4]:
        if _is_valid(cfg):
            start_configs.append(cfg)
    start_configs.append(_make_random_initial())

    time_budget = time_left()
    time_per_refine = min(20.0, time_budget / max(1, len(start_configs)))

    for cfg_init in start_configs:
        if time_left() < 5:
            break
        end_time = time.time() + time_per_refine
        _refine_lbfgs(cfg_init, end_time)

    # --- Phase 3: Simulated Annealing refinement ---
    def _sa_refine(cfg_init, max_time):
        nonlocal best_side, best_cfg, best_center, best_angle

        cfg = [row[:] for row in cfg_init]
        cx, cy, ca = best_center[0], best_center[1], best_angle
        cur_score = _outer_hex_side(cfg, (cx, cy), ca)
        local_best = cur_score
        local_best_cfg = [row[:] for row in cfg]

        T = 0.15
        T_end = 0.001
        alpha = 0.997

        while time.time() < max_time and T > T_end:
            # Perturb one hexagon
            idx = random.randrange(N)
            old = cfg[idx][:]
            cfg[idx][0] += random.gauss(0, 0.1)
            cfg[idx][1] += random.gauss(0, 0.1)
            cfg[idx][2] += random.gauss(0, 2.0)

            if not _is_valid(cfg):
                cfg[idx] = old
                T *= alpha
                continue

            new_score = _outer_hex_side(cfg, (cx, cy), ca)
            if new_score < cur_score or random.random() < math.exp(-(new_score - cur_score) / T):
                cur_score = new_score
                if new_score < local_best:
                    local_best = new_score
                    local_best_cfg = [row[:] for row in cfg]
            else:
                cfg[idx] = old
            T *= alpha

        if local_best < best_side:
            best_side = local_best
            best_cfg = local_best_cfg

    if time_left() > 15:
        _sa_refine(best_cfg, time.time() + min(30.0, time_left() - 5))

    # --- Phase 4: Nelder-Mead final polish on outer hexagon params ---
    if best_cfg is not None and time_left() > 10:
        def _nm_obj(x):
            cx, cy, ca = x
            return _outer_hex_side(best_cfg, (cx, cy), ca)

        res = minimize(_nm_obj, [best_center[0], best_center[1], best_angle],
                       method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14})
        if res.fun < best_side:
            best_side = res.fun
            best_center = (res.x[0], res.x[1])
            best_angle = res.x[2]

    # --- Phase 5: Final fine SA ---
    if time_left() > 8:
        cfg = [row[:] for row in best_cfg]
        cx, cy, ca = best_center[0], best_center[1], best_angle
        cur_score = best_side
        T = 0.03
        T_end = 0.0005
        alpha = 0.995

        while time_left() > 3 and T > T_end:
            idx = random.randrange(N)
            old = cfg[idx][:]
            cfg[idx][0] += random.gauss(0, 0.03)
            cfg[idx][1] += random.gauss(0, 0.03)
            cfg[idx][2] += random.gauss(0, 0.5)

            if not _is_valid(cfg):
                cfg[idx] = old
                T *= alpha
                continue

            new_score = _outer_hex_side(cfg, (cx, cy), ca)
            if new_score < cur_score or random.random() < math.exp(-(new_score - cur_score) / T):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [row[:] for row in cfg]
            else:
                cfg[idx] = old
            T *= alpha

    # Final validation
    if not _is_valid(best_cfg):
        # Fallback to a known-valid configuration
        cfg = _make_hex_tiling_initial()
        for p in cfg:
            p[0] *= 1.15
            p[1] *= 1.15
        best_cfg = cfg
        best_side = _outer_hex_side(cfg, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, best_center, best_side, best_angle


if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
