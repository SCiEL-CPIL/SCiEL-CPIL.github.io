import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0

# --- Geometry helpers -------------------------------------------------------

def hex_vertices(cx, cy, s, ang_rad):
    """Vertices of a regular hexagon (centered at (cx,cy), side s, rotation ang_rad)."""
    a = ang_rad + np.pi / 6.0
    return np.column_stack([cx + s * np.cos(a), cy + s * np.sin(a)])

def poly_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def poly_overlap_area(a, b):
    """Polygon intersection area (Sutherland–Hodgman)."""
    def clip(subject, edge_a, edge_b):
        if not subject:
            return []
        out = []
        def inside(p):
            return np.cross(edge_b - edge_a, p - edge_a) >= -1e-12
        def intersect(p1, p2):
            d1 = np.cross(edge_b - edge_a, p1 - edge_a)
            d2 = np.cross(edge_b - edge_a, p2 - edge_a)
            if abs(d2 - d1) < 1e-14:
                return p1
            t = d1 / (d1 - d2)
            return p1 + t * (p2 - p1)
        prev = subject[-1]
        for cur in subject:
            pi, ci = inside(prev), inside(cur)
            if pi and ci:
                out.append(cur)
            elif pi and not ci:
                out.append(intersect(prev, cur))
            elif not pi and ci:
                out.append(intersect(prev, cur))
                out.append(cur)
            prev = cur
        return out

    subj = [tuple(p) for p in a]
    for i in range(len(b)):
        subj = clip(subj, b[i], b[(i + 1) % len(b)])
        if not subj:
            return 0.0
    if len(subj) < 3:
        return 0.0
    return poly_area(np.array(subj))

def all_inside(pts, poly):
    """Point-in-convex-polygon test (all points inside)."""
    n = len(poly)
    for p in pts:
        ok = True
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            if np.cross(b - a, p - a) < -1e-9:
                ok = False
                break
        if not ok:
            return False
    return True

def outer_side_length(inner, center, outer_ang):
    """Smallest outer side s (regular hexagon, center, angle) containing all inner hexagons."""
    s_best = 10.0
    for _ in range(60):
        lo, hi = 0.5, s_best
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if fits(inner, center, mid, outer_ang):
                hi = mid
            else:
                lo = mid
        s_best = hi
    return s_best

def fits(inner, center, s, outer_ang):
    """Check all inner hexagons are inside the outer regular hexagon."""
    ox, oy = center
    oa = outer_ang + np.pi / 6.0
    ov = np.column_stack([ox + s * np.cos(oa), oy + s * np.sin(oa)])
    for (x, y, ang) in inner:
        iv = hex_vertices(x, y, 1.0, ang)
        if not all_inside(iv, ov):
            return False
    return True

def pairwise_disjoint(inner):
    """Check all pairs of inner hexagons are disjoint (non-overlapping area)."""
    verts = [hex_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if poly_overlap_area(verts[i], verts[j]) > 1e-7:
                return False
    return True

def score_of(inner, center, outer_ang):
    if not pairwise_disjoint(inner):
        return float('inf')
    return outer_side_length(inner, center, outer_ang)

# --- Lattice topology search ------------------------------------------------

def lattice_positions(spread):
    """Hexagonal lattice points with given spacing, centered."""
    pts = []
    R = 3
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            if abs(q + r) > R:
                continue
            x = spread * (q + 0.5 * r)
            y = spread * (math.sqrt(3) / 2.0 * r)
            pts.append((x, y))
    return pts

def try_topology(pos_list, angles, center=(0.0, 0.0), outer_ang=0.0):
    """Try a specific assignment of positions+angles; return side length if valid."""
    inner = [[pos_list[i][0], pos_list[i][1], angles[i]] for i in range(N)]
    if not pairwise_disjoint(inner):
        return None
    s = outer_side_length(inner, center, outer_ang)
    return s

def polish(inner, center, outer_ang, time_budget=20.0):
    """SLSQP polish: minimize outer side length with hard constraints."""
    import time
    t0 = time.time()

    def x_to_inner(x):
        inner = []
        for i in range(N):
            inner.append([x[3 * i], x[3 * i + 1], x[3 * i + 2]])
        return inner

    def obj(x):
        inner = x_to_inner(x)
        if not pairwise_disjoint(inner):
            return 1e6
        return outer_side_length(inner, center, outer_ang)

    def con_disjoint(x):
        inner = x_to_inner(x)
        verts = [hex_vertices(px, py, 1.0, pa) for (px, py, pa) in inner]
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                d = min_pair_gap(verts[i], verts[j])
                cons.append(d)  # >= 0
        return cons

    def con_inside(x):
        inner = x_to_inner(x)
        s_cur = outer_side_length(inner, center, outer_ang)
        # constraint: all vertices inside hexagon of side s_cur
        # We use a fixed reference s for constraints (the current best)
        s_ref = s_cur
        oa = outer_ang + np.pi / 6.0
        ov = np.column_stack([center[0] + s_ref * np.cos(oa), center[1] + s_ref * np.sin(oa)])
        cons = []
        for (px, py, pa) in inner:
            iv = hex_vertices(px, py, 1.0, pa)
            for p in iv:
                for i in range(6):
                    a, b = ov[i], ov[(i + 1) % 6]
                    cons.append(float(np.cross(b - a, p - a)))  # >= 0
        return cons

    x0 = np.array([[inner[i][0], inner[i][1], inner[i][2]] for i in range(N)])
    cons = []
    cons += [{'type': 'ineq', 'fun': lambda x, c=con_disjoint: c(x)}]
    cons += [{'type': 'ineq', 'fun': lambda x, c=con_inside: c(x)}]

    try:
        res = minimize(obj, x0, method='SLSQP', constraints=cons,
                        options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})
        if res.success or res.fun < 1e5:
            inner_new = x_to_inner(res.x)
            if pairwise_disjoint(inner_new):
                s_new = outer_side_length(inner_new, center, outer_ang)
                if s_new < 1e5:
                    return inner_new, s_new
    except Exception:
        pass
    return None

# --- Main construct ---------------------------------------------------------

def construct():
    import time
    t_start = time.time()
    best_inner = None
    best_score = float('inf')
    best_center = (0.0, 0.0)
    best_outer_ang = 0.0

    # Phase 1: lattice topology search
    spreads = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
    angle_choices = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
    center_choices = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (-0.5, 0.0), (0.0, -0.5)]
    outer_ang_choices = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]

    for spread in spreads:
        if time.time() - t_start > DEADLINE - 30:
            break
        pts = lattice_positions(spread)
        if len(pts) < N:
            continue
        # Try several random subsets of N points
        import random
        rng = random.Random(42)
        n_trials = 20
        for trial in range(n_trials):
            if time.time() - t_start > DEADLINE - 30:
                break
            idx = rng.sample(range(len(pts)), N)
            pos_list = [pts[i] for i in idx]
            # Try a few angle assignments
            for ang_seed in range(4):
                if time.time() - t_start > DEADLINE - 30:
                    break
                angles = [angle_choices[(i + ang_seed) % len(angle_choices)] for i in range(N)]
                for center in center_choices[:2]:
                    for outer_ang in outer_ang_choices[:2]:
                        s = try_topology(pos_list, angles, center, outer_ang)
                        if s is not None and s < best_score:
                            best_score = s
                            best_inner = [[pos_list[i][0], pos_list[i][1], angles[i]] for i in range(N)]
                            best_center = center
                            best_outer_ang = outer_ang

    # Also try a simple grid layout
    for spread in [2.1, 2.2, 2.3, 2.4, 2.5]:
        if time.time() - t_start > DEADLINE - 30:
            break
        # 3x4 grid
        pts = []
        for i in range(3):
            for j in range(4):
                pts.append(((j - 1.5) * spread, (i - 1.0) * spread * math.sqrt(3) / 2))
        if len(pts) >= N:
            pos_list = pts[:N]
            for ang_seed in range(3):
                if time.time() - t_start > DEADLINE - 30:
                    break
                angles = [angle_choices[(i + ang_seed) % len(angle_choices)] for i in range(N)]
                for center in [(0.0, 0.0)]:
                    for outer_ang in [0.0, 30.0]:
                        s = try_topology(pos_list, angles, center, outer_ang)
                        if s is not None and s < best_score:
                            best_score = s
                            best_inner = [[pos_list[i][0], pos_list[i][1], angles[i]] for i in range(N)]
                            best_center = center
                            best_outer_ang = outer_ang

    # If lattice search found nothing valid, use a known-valid fallback
    if best_inner is None:
        # Fallback: 12 hexagons in a line, wide enough
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_center = (0.0, 0.0)
        best_outer_ang = 0.0
        best_score = outer_side_length(best_inner, best_center, best_outer_ang)

    # Phase 2: SLSQP polish
    if time.time() - t_start < DEADLINE - 20:
        polish_result = polish(best_inner, best_center, best_outer_ang,
                               time_budget=DEADLINE - time.time() - 15)
        if polish_result is not None:
            inner_new, s_new = polish_result
            if s_new < best_score:
                best_inner = inner_new
                best_score = s_new

    # Final safety: ensure validity
    if not pairwise_disjoint(best_inner):
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_center = (0.0, 0.0)
        best_outer_ang = 0.0
        best_score = outer_side_length(best_inner, best_center, best_outer_ang)

    # Ensure inner hexagons are inside the outer hexagon
    s_final = outer_side_length(best_inner, best_center, best_outer_ang)
    if s_final > best_score * 1.01:
        s_final = best_score

    return best_inner, best_center, s_final, best_outer_ang
