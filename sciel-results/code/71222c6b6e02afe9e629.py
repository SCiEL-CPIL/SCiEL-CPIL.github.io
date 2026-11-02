"""Unit regular hexagons inside a regular hexagon — lattice + smooth L-BFGS-B polish."""
import math
import time

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 118.0  # safety margin within 150 s limit

# ---------- geometry helpers ----------

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation (degrees)."""
    ar = math.radians(angle_deg)
    return [(cx + math.cos(ar + math.pi * k / 3.0),
             cy + math.sin(ar + math.pi * k / 3.0)) for k in range(6)]


def _min_dist_poly(poly1, poly2):
    """Minimum Euclidean distance between two convex polygons (0 if they intersect)."""
    # Quick bbox check
    xs1 = [p[0] for p in poly1]; ys1 = [p[1] for p in poly1]
    xs2 = [p[0] for p in poly2]; ys2 = [p[1] for p in poly2]
    if max(xs1) < min(xs2) - 1e-12 or max(xs2) < min(xs1) - 1e-12 or \
       max(ys1) < min(ys2) - 1e-12 or max(ys2) < min(ys1) - 1e-12:
        # disjoint bboxes: min distance is min of vertex-to-vertex distances
        best = float('inf')
        for a in poly1:
            for b in poly2:
                d = (a[0]-b[0])**2 + (a[1]-b[1])**2
                if d < best:
                    best = d
        return math.sqrt(best)
    # Check if they overlap (vertex-in-polygon or edge intersection)
    if _polygons_overlap(poly1, poly2):
        return 0.0
    # Compute min distance via vertex-to-vertex and vertex-to-edge
    best = float('inf')
    for a in poly1:
        for b in poly2:
            d = (a[0]-b[0])**2 + (a[1]-b[1])**2
            if d < best:
                best = d
        # vertex a to each edge of poly2
        for i in range(6):
            d = _pt_seg_dist(a, poly2[i], poly2[(i+1) % 6])
            if d < best:
                best = d
    for b in poly2:
        for i in range(6):
            d = _pt_seg_dist(b, poly1[i], poly1[(i+1) % 6])
            if d < best:
                best = d
    return math.sqrt(best)


def _pt_seg_dist(pt, a, b):
    """Squared distance from point pt to segment ab."""
    ax, ay = a; bx, by = b; px, py = pt
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return (px - ax)**2 + (py - ay)**2
    t = ((px - ax)*dx + (py - ay)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return (px - cx)**2 + (py - cy)**2


def _polygons_overlap(poly1, poly2, eps=1e-9):
    """Check if two convex hexagons have overlapping interiors."""
    for v in poly1:
        if _point_in_convex_polygon(v, poly2, eps):
            return True
    for v in poly2:
        if _point_in_convex_polygon(v, poly1, eps):
            return True
    for i in range(6):
        for j in range(6):
            if _segments_intersect(poly1[i], poly1[(i+1) % 6],
                                   poly2[j], poly2[(j+1) % 6], eps):
                return True
    return False


def _point_in_convex_polygon(point, polygon, eps=1e-9):
    px, py = point
    for i in range(6):
        ax, ay = polygon[i]
        bx, by = polygon[(i+1) % 6]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > eps:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4, eps=1e-9):
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    if abs(d1) <= eps and _on_segment(p3, p1, p4, eps): return True
    if abs(d2) <= eps and _on_segment(p3, p2, p4, eps): return True
    if abs(d3) <= eps and _on_segment(p1, p3, p2, eps): return True
    if abs(d4) <= eps and _on_segment(p1, p4, p2, eps): return True
    return False


def _on_segment(p, q, r, eps=1e-9):
    if (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and
        min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps and
        abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])) <= eps):
        return True
    return False

# ---------- outer hexagon (LP) ----------

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """LP: find minimum side length of outer hexagon with given angle containing all inner hex vertices."""
    from scipy.optimize import linprog

    all_verts = []
    for ix, iy, ia in inner_hexes:
        all_verts.extend(_hex_vertices(ix, iy, ia))

    c = [0.0, 0.0, 1.0]  # minimize t (inradius)
    A_ub = []
    b_ub = []

    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        for vx, vy in all_verts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
        for vx, vy in all_verts:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
    if result.success:
        cx, cy, t = result.x
        return cx, cy, t / (math.sqrt(3) / 2)
    # fallback
    cx = sum(v[0] for v in all_verts) / len(all_verts)
    cy = sum(v[1] for v in all_verts) / len(all_verts)
    req = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        for vx, vy in all_verts:
            d = (vx - cx) * nx + (vy - cy) * ny
            req = max(req, abs(d))
    return cx, cy, req / (math.sqrt(3) / 2)


def _min_enclosing_hex_side(inner_hexes):
    """Minimize outer hex side over angle (coarse sweep)."""
    best = (float('inf'), 0.0, 0.0, 0.0)
    for ang in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]:
        cx, cy, s = _min_enclosing_hex_side_lp(inner_hexes, ang)
        if s < best[0]:
            best = (s, cx, cy, ang)
    return best[1], best[2], best[0], best[3]

# ---------- smooth objective for L-BFGS-B ----------

def _soft_nonoverlap_penalty(inner_hexes, scale=50.0):
    """Smooth penalty: sum over pairs of max(0, gap_target - min_dist)^2 via soft approximation.
    For two unit hexagons with same orientation, minimum center distance for non-overlap is sqrt(3).
    We use the actual polygon min-distance for accuracy."""
    n = len(inner_hexes)
    polys = [_hex_vertices(h[0], h[1], h[2]) for h in inner_hexes]
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_dist_poly(polys[i], polys[j])
            # We want d >= 0 (non-overlap). Penalize overlap softly.
            # Use a smooth hinge: if d < 0.001, penalty grows.
            deficit = 0.001 - d
            if deficit > 0:
                penalty += deficit * deficit
    return penalty


def _smooth_objective(params):
    """L-BFGS-B objective: outer side + smooth non-overlap penalty.
    params: [x1,y1,a1, ..., x11,y11,a11] (33 values) or with shared angle (23 values)."""
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        return 9999.0

    n = N
    # Determine if shared angle or per-hex angle
    if len(params) == 3 * n:
        inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(n)]
    else:
        # shared angle: last element
        shared_a = params[-1]
        inner = [[params[2*i], params[2*i+1], shared_a] for i in range(n)]

    # Quick non-overlap check (cheap center-distance heuristic first)
    min_cdist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            cd = math.sqrt(dx*dx + dy*dy)
            if cd < min_cdist:
                min_cdist = cd
    # For same-orientation hexagons, centers must be >= sqrt(3) apart (approximately)
    if min_cdist < math.sqrt(3) - 0.05:
        # Likely overlapping; compute exact penalty
        pen = _soft_nonoverlap_penalty(inner)
        if pen > 1e-6:
            # Outer side estimate + heavy penalty
            cx = sum(h[0] for h in inner) / n
            cy = sum(h[1] for h in inner) / n
            maxr = max(math.sqrt((h[0]-cx)**2 + (h[1]-cy)**2) + 1.0 for h in inner)
            outer = maxr / (math.sqrt(3)/2)
            return outer + 500.0 * pen + 50.0 * max(0, math.sqrt(3) - min_cdist)

    # Compute outer hex side (LP with a few angles)
    best_s = float('inf')
    for ang in [0.0, 15.0, 30.0, 45.0]:
        _, _, s = _min_enclosing_hex_side_lp(inner, ang)
        if s < best_s:
            best_s = s

    # Add a small smooth penalty for any near-overlap
    pen = _soft_nonoverlap_penalty(inner)
    return best_s + 200.0 * pen


# ---------- L-BFGS-B optimization ----------

def _lbgfsb_optimize(initial_inner, shared_angle=True, max_iter=300):
    """Run L-BFGS-B to optimize positions (and optionally shared angle)."""
    from scipy.optimize import minimize

    n = N
    if shared_angle:
        # 2*n positions + 1 shared angle = 23
        x0 = []
        for h in initial_inner:
            x0.append(h[0])
            x0.append(h[1])
        x0.append(initial_inner[0][2])
        bounds = [(None, None)] * (2 * n) + [(-180.0, 180.0)]
        # Objective for shared-angle case
        def obj(p):
            if time.time() > _DEADLINE - 5.0:
                return 9999.0
            inner = [[p[2*i], p[2*i+1], p[-1]] for i in range(n)]
            # Quick check
            min_cd = float('inf')
            for i in range(n):
                for j in range(i+1, n):
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    cd = math.sqrt(dx*dx + dy*dy)
                    if cd < min_cd:
                        min_cd = cd
            if min_cd < math.sqrt(3) - 0.1:
                pen = _soft_nonoverlap_penalty(inner)
                cx = sum(h[0] for h in inner) / n
                cy = sum(h[1] for h in inner) / n
                maxr = max(math.sqrt((h[0]-cx)**2 + (h[1]-cy)**2) + 1.0 for h in inner)
                outer = maxr / (math.sqrt(3)/2)
                return outer + 500.0 * pen + 50.0 * max(0, math.sqrt(3) - 0.01 - min_cd)
            best_s = float('inf')
            for ang in [0.0, 15.0, 30.0, 45.0]:
                _, _, s = _min_enclosing_hex_side_lp(inner, ang)
                if s < best_s:
                    best_s = s
            pen = _soft_nonoverlap_penalty(inner)
            return best_s + 200.0 * pen

        result = minimize(obj, x0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
                          bounds=bounds)
        if result.success and result.fun < 100.0:
            polished = [[result.x[2*i], result.x[2*i+1], result.x[-1]] for i in range(n)]
            return polished
    else:
        # 3*n = 33 variables
        x0 = []
        for h in initial_inner:
            x0.extend([h[0], h[1], h[2]])
        bounds = [(None, None)] * (3 * n)
        result = minimize(_smooth_objective, x0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
                          bounds=bounds)
        if result.success and result.fun < 100.0:
            polished = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(n)]
            return polished

    return initial_inner


# ---------- final high-precision angle sweep ----------

def _fine_angle_sweep(inner_hexes, base_angle, n_steps=40, span=20.0):
    """Sweep angles around base_angle finely."""
    best_s = float('inf')
    best_cx, best_cy, best_ang = 0.0, 0.0, base_angle
    for k in range(n_steps):
        ang = (base_angle + span * (k / (n_steps - 1) - 0.5)) % 60
        cx, cy, s = _min_enclosing_hex_side_lp(inner_hexes, ang)
        if s < best_s:
            best_s = s
            best_cx, best_cy, best_ang = cx, cy, ang
    return best_cx, best_cy, best_s, best_ang


# ---------- construct ----------

def construct():
    global _DEADLINE

    if time.time() > _DEADLINE - 3.0:
        # Emergency fallback
        inner = [[0.0, 0.0, 30.0]] * N
        return inner, (0.0, 0.0), 50.0, 0.0

    sqrt3 = math.sqrt(3)
    scale = 1.002  # tiny gap to ensure strict disjointness initially

    # Build the 11-point hexagonal cluster on triangular lattice
    # Points: (i,j) in axial coords -> position = i*(sqrt3,0) + j*(sqrt3/2, 1.5)
    # We want the 11 points closest to origin: center + 6 ring + 4 from second ring
    # Axial coords for a hex cluster of radius 1 (7 points) + 4 more
    # Let's enumerate axial coords and pick 11 closest
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * sqrt3 * scale + j * sqrt3 / 2 * scale
            y = j * 1.5 * scale
            d2 = x * x + y * y
            lattice_points.append((d2, x, y, i, j))
    lattice_points.sort(key=lambda t: t[0])

    # Deduplicate and take 11
    seen = set()
    positions = []
    for d2, x, y, i, j in lattice_points:
        key = (round(x, 6), round(y, 6))
        if key in seen:
            continue
        seen.add(key)
        positions.append((x, y))
        if len(positions) == N:
            break

    if len(positions) < N:
        # Fallback: just place them in a row
        positions = [(k * sqrt3 * scale, 0.0) for k in range(N)]

    # All hexagons at 30 degrees (pointy-top) — aligns with lattice normals
    initial_inner = [[px, py, 30.0] for px, py in positions]

    # Compute initial enclosing hex
    cx, cy, side, angle = _min_enclosing_hex_side(initial_inner)
    best_side = side
    best_inner = [h[:] for h in initial_inner]
    best_center = (cx, cy)
    best_angle = angle

    # --- Multi-start L-BFGS-B optimization ---
    if time.time() < _DEADLINE - 25.0:
        # Start 1: shared angle optimization from lattice
        polished = _lbgfsb_optimize(initial_inner, shared_angle=True, max_iter=250)
        if polished != initial_inner:
            pcx, pcy, pside, pang = _min_enclosing_hex_side(polished)
            if pside < best_side - 1e-8:
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_angle = pang

    # Start 2: per-hex angle (33 vars) if time permits
    if time.time() < _DEADLINE - 20.0:
        polished2 = _lbgfsb_optimize(best_inner, shared_angle=False, max_iter=150)
        if polished2 != best_inner:
            pcx, pcy, pside, pang = _min_enclosing_hex_side(polished2)
            if pside < best_side - 1e-8:
                best_side = pside
                best_inner = polished2
                best_center = (pcx, pcy)
                best_angle = pang

    # Start 3: slight perturbation multi-start
    if time.time() < _DEADLINE - 15.0:
        import random
        random.seed(42)
        for trial in range(3):
            perturbed = []
            for h in best_inner:
                px = h[0] + random.uniform(-0.05, 0.05)
                py = h[1] + random.uniform(-0.05, 0.05)
                pa = h[2] + random.uniform(-2.0, 2.0)
                perturbed.append([px, py, pa])
            p3 = _lbgfsb_optimize(perturbed, shared_angle=True, max_iter=100)
            if p3 != perturbed:
                pcx, pcy, pside, pang = _min_enclosing_hex_side(p3)
                if pside < best_side - 1e-8:
                    best_side = pside
                    best_inner = p3
                    best_center = (pcx, pcy)
                    best_angle = pang
            if time.time() > _DEADLINE - 12.0:
                break

    # --- Final fine angle sweep ---
    if time.time() < _DEADLINE - 8.0:
        fcx, fcy, fside, fang = _fine_angle_sweep(best_inner, best_angle % 60, n_steps=50, span=30.0)
        if fside < best_side - 1e-10:
            best_side = fside
            best_center = (fcx, fcy)
            best_angle = fang

    # Even finer sweep around the found optimum
    if time.time() < _DEADLINE - 5.0:
        fcx, fcy, fside, fang = _fine_angle_sweep(best_inner, best_angle % 60, n_steps=30, span=5.0)
        if fside < best_side - 1e-10:
            best_side = fside
            best_center = (fcx, fcy)
            best_angle = fang

    return best_inner, (best_center[0], best_center[1]), best_side, best_angle
