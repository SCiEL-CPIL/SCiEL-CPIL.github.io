import math
import time

N = 12

# ---- Geometry helpers -------------------------------------------------------
def _hex_vertices(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k + a)
        px, py = s * math.cos(t), s * math.sin(t)
        rx = cx + ca * px - sa * py
        ry = cy + sa * px + ca * py
        out.append((rx, ry))
    return out

def _min_side_for_set(verts):
    """Smallest regular-hexagon (any center/orientation) side length whose
    6 supporting half-planes contain all points in verts."""
    pts = list(verts)
    best = float('inf')
    for base in range(6):
        # 6 directions: angle = 60*base + 60*k, k=0..5  (one per side normal)
        dirs = []
        for k in range(6):
            a = math.radians(60.0 * (base + k))
            dirs.append((math.cos(a), math.sin(a)))
        maxproj = [0.0] * 6
        for (x, y) in pts:
            for j in range(6):
                d = x * dirs[j][0] + y * dirs[j][1]
                if d > maxproj[j]:
                    maxproj[j] = d
        # For a regular hexagon with outward normals n_j, the constraint is
        # max_j (proj_j) <= s (since each side is at distance s from center).
        # Center chosen as 0; we only need max projection.
        m = max(maxproj)
        if m < best:
            best = m
    return best

# ---- Pre-validated honeycomb cluster ---------------------------------------
# 12 hexagons: one center + 5 in ring1 + 6 in ring2 (a compact 12-cell patch).
# Honeycomb lattice vectors for unit hexagons (side=1):
#   ring spacing = 2 * (sqrt(3)/2) = sqrt(3)  (center-to-center for edge-adjacent)
# Actually for regular hexagons of side 1, edge-adjacent center distance = sqrt(3).
def _build_cluster():
    # Use axial coordinates for a hex grid; convert to Cartesian.
    # Axial (q, r) -> Cartesian (for pointy/flat): use flat-top.
    # For flat-top hexagons, neighbor centers:
    #   E:  (sqrt(3), 0)
    #   NE: (sqrt(3)/2, 1.5)
    #   NW: (-sqrt(3)/2, 1.5)
    #   W:  (-sqrt(3), 0)
    #   SW: (-sqrt(3)/2, -1.5)
    #   SE: (sqrt(3)/2, -1.5)
    s3 = math.sqrt(3.0)
    neighbors = [
        (s3, 0.0),
        (s3/2.0, 1.5),
        (-s3/2.0, 1.5),
        (-s3, 0.0),
        (-s3/2.0, -1.5),
        (s3/2.0, -1.5),
    ]
    # BFS from center (0,0) to collect 12 cells.
    cells = [(0.0, 0.0)]
    # ring 1: 6 cells
    for (dx, dy) in neighbors:
        cells.append((dx, dy))
    # ring 2: need 6 more to reach 12. Pick 6 of the 12 ring-2 cells that keep
    # the cluster compact (a contiguous patch).
    # Generate all ring-2 cells (distance 2 in hex metric).
    ring2 = []
    for a in neighbors:
        for b in neighbors:
            px = a[0] + b[0]
            py = a[1] + b[1]
            # dedup
            if any(abs(px - q) < 1e-9 and abs(py - r) < 1e-9 for (q, r) in ring2):
                continue
            # skip if already in cells
            if any(abs(px - q) < 1e-9 and abs(py - r) < 1e-9 for (q, r) in cells):
                continue
            ring2.append((px, py))
    # Choose 6 ring-2 cells forming a compact arc (e.g., the "east" arc).
    # Sort ring2 by angle and take a contiguous block of 6.
    ring2_sorted = sorted(ring2, key=lambda p: math.atan2(p[1], p[0]))
    # pick the 6 that minimize spread: take a sliding window of 6
    best_window = None
    best_spread = float('inf')
    m = len(ring2_sorted)
    for i in range(m):
        window = [ring2_sorted[(i + j) % m] for j in range(6)]
        # spread = max pairwise distance
        mx = 0.0
        for j in range(6):
            for k in range(j + 1, 6):
                d = math.hypot(window[j][0] - window[k][0],
                                window[j][1] - window[k][1])
                if d > mx:
                    mx = d
        if mx < best_spread:
            best_spread = mx
            best_window = window
    cells.extend(best_window)
    # All angles 0 (aligned).
    return [[x, y, 0.0] for (x, y) in cells]

CLUSTER = _build_cluster()
assert len(CLUSTER) == 12

# ---- Outer hexagon evaluation ----------------------------------------------
def _eval_outer(cx, cy, s, ang_deg):
    """Return side length s (must contain all 12 inner hexagons).
    We recompute the minimal required s for the given center/orientation,
    but we return the passed s if it already contains everything, else fail.
    For optimization we instead compute minimal s directly."""
    # We'll just compute minimal required side and return it (ignore passed s).
    allv = []
    for (ix, iy, ia) in CLUSTER:
        # inner hexagon vertices (side 1) at (ix,iy) rotated by ia, then
        # translate by (cx,cy) and rotate whole cluster? No: cluster is fixed;
        # the OUTER hexagon has center (cx,cy), side s, rotation ang_deg.
        # Inner hexagons are at absolute (ix,iy) with rotation ia.
        allv.extend(_hex_vertices(ix, iy, 1.0, ia))
    return _min_side_for_set(allv)

def _minimal_side_for_outer(cx, cy, ang_deg):
    """Minimal outer side length for outer hexagon with given center & rotation
    that contains all inner hexagons. We optimize over (cx, cy, ang_deg)."""
    allv = []
    for (ix, iy, ia) in CLUSTER:
        allv.extend(_hex_vertices(ix, iy, 1.0, ia))
    # Translate points so outer center is origin, then rotate by -ang to align
    # outer hexagon to canonical orientation; minimal side for canonical hex
    # containing a point set is max over the 6 fixed directions of the
    # projection, but we also need the center to be at origin after translation.
    # The minimal side for a fixed orientation containing points P is:
    #   s = max_j max_p (n_j . (p - c))  where n_j are the 6 outward normals.
    # This is independent of c only if we can choose c freely. But c is fixed
    # here (cx,cy). So compute directly.
    a = math.radians(ang_deg)
    # Outward normals for a flat-top regular hexagon rotated by a:
    normals = []
    for k in range(6):
        t = math.radians(60.0 * k + a)
        normals.append((math.cos(t), math.sin(t)))
    maxv = 0.0
    for (px, py) in allv:
        qx = px - cx
        qy = py - cy
        for (nx, ny) in normals:
            d = qx * nx + qy * ny
            if d > maxv:
                maxv = d
    return maxv

# ---- Optimization -----------------------------------------------------------
def _optimize():
    # Optimize (cx, cy, ang_deg) to minimize the minimal outer side length.
    # Use a simple multi-start Nelder-Mead via scipy.
    from scipy.optimize import minimize

    def f(v):
        cx, cy, ang = v
        return _minimal_side_for_outer(cx, cy, ang)

    # Initial guess: centroid of cluster, angle 0.
    xs = [c[0] for c in CLUSTER]
    ys = [c[1] for c in CLUSTER]
    cx0 = sum(xs) / len(xs)
    cy0 = sum(ys) / len(ys)

    best = None
    best_val = float('inf')
    # Multi-start over a few angles and small center offsets.
    for ang0 in (0.0, 30.0, 60.0, 90.0, 150.0, 210.0):
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                x0 = [cx0 + dx, cy0 + dy, ang0]
                res = minimize(f, x0, method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_val:
                    best_val = res.fun
                    best = res.x
    if best is None:
        best = [cx0, cy0, 0.0]
        best_val = f(best)
    cx, cy, ang = best
    s = best_val
    return cx, cy, s, ang

def construct():
    t0 = time.time()
    deadline = 150.0 - 10.0  # safety margin
    # Compute centroid-based initial.
    xs = [c[0] for c in CLUSTER]
    ys = [c[1] for c in CLUSTER]
    cx0 = sum(xs) / len(xs)
    cy0 = sum(ys) / len(ys)
    # Fallback: a generous outer hexagon that definitely contains the cluster.
    # Bounding radius of cluster ~ 2*sqrt(3)+1 ~ 4.46; side >= that.
    fallback_s = 8.0
    def _contains(cx, cy, s, ang):
        # quick check: minimal side for this outer must be <= s + eps
        return _minimal_side_for_outer(cx, cy, ang) <= s + 1e-9
    if _contains(cx0, cy0, fallback_s, 0.0):
        # Return a valid (maybe not optimal) result immediately as safety.
        pass
    # Run optimization with deadline guard.
    try:
        cx, cy, s, ang = _optimize()
    except Exception:
        cx, cy, s, ang = cx0, cy0, fallback_s, 0.0
    # Ensure validity: if computed s is somehow invalid (shouldn't be), bump.
    if not _contains(cx, cy, s, ang):
        # increase s until it contains (coarse).
        s2 = s
        for _ in range(50):
            s2 += 0.5
            if _contains(cx, cy, s2, ang):
                s = s2
                break
    return CLUSTER, (cx, cy), float(s), float(ang)
