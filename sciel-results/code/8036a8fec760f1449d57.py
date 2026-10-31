"""Pack 11 unit hexagons in a minimal regular hexagon via continuous optimization."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

# ---------- geometry helpers ----------
def hexagon_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.deg2rad(angle_deg)
    verts = []
    for k in range(6):
        t = a + k * np.pi / 3.0
        verts.append((cx + np.cos(t), cy + np.sin(t)))
    return np.array(verts)

def poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygons_overlap(A, B):
    """Return overlap area of two convex polygons (>=0)."""
    n, m = len(A), len(B)
    # Sutherland–Hodgman clip A by each edge of B
    output = A
    for i in range(m):
        if len(output) == 0:
            return 0.0
        edge_start = B[i]
        edge_end = B[(i + 1) % m]
        edge = edge_end - edge_start
        # inside test: cross(edge, p - edge_start) >= 0
        def inside(p):
            return np.cross(edge, p - edge_start) >= -1e-12
        input_pts = output
        output = []
        n_in = len(input_pts)
        for j in range(n_in):
            cur = input_pts[j]
            nxt = input_pts[(j + 1) % n_in]
            cur_in = inside(cur)
            nxt_in = inside(nxt)
            if cur_in:
                output.append(cur)
                if not nxt_in:
                    # exit: intersect
                    d = nxt - cur
                    denom = np.cross(edge, d)
                    if abs(denom) > 1e-15:
                        t = np.cross(edge_start - cur, d) / denom
                        output.append(cur + t * d)
            else:
                if nxt_in:
                    # enter: intersect
                    d = nxt - cur
                    denom = np.cross(edge, d)
                    if abs(denom) > 1e-15:
                        t = np.cross(edge_start - cur, d) / denom
                        output.append(cur + t * d)
    if len(output) < 3:
        return 0.0
    return poly_area(np.array(output))

def pairwise_overlap(p):
    """p: (3N,) -> [x0,y0,a0, x1,y1,a1, ...] in degrees. Returns total overlap area."""
    if p.size < 3 * N:
        return 1e12
    total = 0.0
    verts = []
    for i in range(N):
        verts.append(hexagon_vertices(p[3*i], p[3*i+1], p[3*i+2]))
    for i in range(N):
        for j in range(i+1, N):
            ov = polygons_overlap(verts[i], verts[j])
            total += ov
    return total

def overlap_penalty(p):
    ov = pairwise_overlap(p)
    if ov <= 1e-9:
        return 0.0
    return ov * 10.0 + min(ov, 1.0) * 100.0

# ---------- minimal enclosing regular hexagon ----------
def min_enclosing_hex(side_points):
    """Given convex-ish set of points (outer boundary of cluster), find min side length
    of a regular hexagon (any rotation, any center) containing all points.
    Returns (side, cx, cy, angle_deg)."""
    # Candidate directions for the hexagon edges: sample rotations 0..60 deg.
    # For a fixed rotation, the minimal hexagon is determined by the support function.
    # A regular hexagon with center c, rotation a, side s contains point p iff for all 6
    # outward normals n_k: n_k . (p - c) <= s * cos(0) ... actually apothem = s*sqrt(3)/2.
    # Constraint: n_k . (p - c) <= apothem = s*sqrt(3)/2 for all k, p.
    # => c must satisfy: for each k, max_p n_k.p - apothem <= n_k.c <= min_p n_k.p + apothem.
    # Feasible if for all k: max_p n_k.p - min_p n_k.p <= 2*apothem = s*sqrt(3).
    # So minimal s for rotation a is max_k (range_k)/sqrt(3).
    best = (None, None, None, None)
    best_side = np.inf
    for a_deg in np.linspace(0, 60, 121):
        a = np.deg2rad(a_deg)
        # outward normals of hexagon edges (6 directions)
        normals = []
        for k in range(6):
            t = a + k * np.pi / 3.0 + np.pi / 2.0
            normals.append(np.array([np.cos(t), np.sin(t)]))
        s_for_this = 0.0
        for n in normals:
            proj = side_points @ n
            rng = proj.max() - proj.min()
            s_for_this = max(s_for_this, rng / np.sqrt(3.0))
        if s_for_this < best_side - 1e-9:
            # find a center c that works: midpoint of ranges works for each axis
            # choose c as average of per-axis midpoints
            c = np.zeros(2)
            cnt = 0
            for n in normals:
                proj = side_points @ n
                mid = (proj.max() + proj.min()) / 2.0
                # c projection on n should be mid; solve least squares over normals
                c += n * mid
                cnt += 1
            c = c / cnt
            best_side = s_for_this
            best = (best_side, float(c[0]), float(c[1]), float(a_deg))
    return best

def cluster_boundary(p):
    """Return outer boundary points of the cluster of hexagons."""
    pts = []
    for i in range(N):
        v = hexagon_vertices(p[3*i], p[3*i+1], p[3*i+2])
        pts.append(v)
    return np.vstack(pts)

# ---------- objective: compact cluster + penalize overlap ----------
def objective(p):
    ov_pen = overlap_penalty(p)
    # diameter-ish: max pairwise center distance to encourage compactness
    if p.size < 3 * N:
        return 1e12
    centers = p.reshape(N, 3)[:, :2]
    # use max distance to centroid (lighter than all pairs)
    c0 = centers.mean(axis=0)
    d = np.max(np.linalg.norm(centers - c0, axis=1))
    return ov_pen + d

# ---------- initial guess ----------
def initial_guess():
    # hexagonal arrangement of 11 hexagons (center + 6 + 4)
    pos = [(0.0, 0.0)]
    for k in range(6):
        t = k * np.pi / 3.0
        pos.append((2.0 * np.cos(t), 2.0 * np.sin(t)))
    # second ring partial
    for k in range(6):
        t = k * np.pi / 3.0 + np.pi / 6.0
        pos.append((2.0 * np.sqrt(3) * np.cos(t), 2.0 * np.sqrt(3) * np.sin(t)))
    # take first N
    pos = pos[:N]
    p = np.zeros(3 * N)
    for i, (x, y) in enumerate(pos):
        p[3*i] = x
        p[3*i+1] = y
        p[3*i+2] = 0.0
    return p

def construct():
    t0 = time.time()
    best_p = initial_guess()
    best_obj = objective(best_p)

    # Phase 1: differential evolution for global compactness
    bounds = [(-10, 10)] * N + [(-10, 10)] * N + [(-30, 30)] * N
    def de_obj(p):
        return objective(p)
    try:
        res = differential_evolution(
            de_obj, bounds,
            seed=42, maxiter=60, popsize=15,
            tol=1e-4, polish=True,
            update_interval=1,
            callback=lambda p, conv: (None),
        )
        if time.time() - t0 > DEADLINE - 20:
            pass
        if res.fun < best_obj:
            best_obj = res.fun
            best_p = res.x
    except Exception:
        pass

    # Phase 2: local refinement (Nelder-Mead / L-BFGS) to reduce overlap to ~0
    def local_obj(p):
        return objective(p)
    try:
        res2 = minimize(local_obj, best_p, method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res2.fun < best_obj:
            best_obj = res2.fun
            best_p = res2.x
    except Exception:
        pass

    # If still overlap, do a few more local passes
    for _ in range(3):
        if time.time() - t0 > DEADLINE - 15:
            break
        ov = pairwise_overlap(best_p)
        if ov < 1e-6:
            break
        try:
            res3 = minimize(local_obj, best_p, method='Powell',
                            options={'maxiter': 1000, 'xtol': 1e-7, 'ftol': 1e-7})
            if res3.fun < best_obj:
                best_obj = res3.fun
                best_p = res3.x
        except Exception:
            break

    # Compute minimal enclosing hexagon
    try:
        boundary = cluster_boundary(best_p)
        side, cx, cy, ang = min_enclosing_hex(boundary)
    except Exception:
        side, cx, cy, ang = 10.0, 0.0, 0.0, 0.0

    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(best_p[3*i]), float(best_p[3*i+1]), float(best_p[3*i+2])])

    # Safety: if deadline close, return
    if time.time() - t0 > DEADLINE:
        pass

    return inner, (cx, cy), side, ang
