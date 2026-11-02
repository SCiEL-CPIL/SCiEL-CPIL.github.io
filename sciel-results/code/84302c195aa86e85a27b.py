"""Honeycomb-topology affine packing of 12 unit hexagons in a regular hexagon."""
import math
import time

N = 12
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill
_HEX_R = 1.0  # circumradius of unit side-length hexagon
_A60 = math.cos(math.pi / 6)
_SQ3 = math.sqrt(3.0)
# honeycomb cell offsets (axial coords q,r)
_AXIAL = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1),
          (-1, 1), (0, 2), (1, 2), (2, 2), (3, 2), (3, 3)]


def _hex_corners(cx, cy, ang):
    """Vertices of a regular hexagon (circumradius _HEX_R) at (cx,cy), rotated by ang rad."""
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for i in range(6):
        a = ang + math.pi / 3 * i
        x = cx + _HEX_R * math.cos(a)
        y = cy + _HEX_R * math.sin(a)
        pts.append((x * ca - y * sa, x * sa + y * ca))
    return pts


def _seg_seg(p1, p2, p3, p4):
    """True if segments p1p2 and p3p4 intersect (proper or touching)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(v) < 1e-9:
            return 0
        return 1 if v > 0 else -1
    def on_seg(a, b, c):
        return (min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9 and
                min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9)
    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_seg(p1, p3, p2):
        return True
    if o2 == 0 and on_seg(p1, p4, p2):
        return True
    if o3 == 0 and on_seg(p3, p1, p4):
        return True
    if o4 == 0 and on_seg(p3, p2, p4):
        return True
    return False


def _hex_overlap(a, b):
    """True if two hexagons (corner lists) intersect (shared edge counts as overlap)."""
    for i in range(6):
        for j in range(6):
            if _seg_seg(a[i], a[(i + 1) % 6], b[j], b[(j + 1) % 6]):
                return True
    return False


def _build_inner(s, sh, rot):
    """Return list of (x, y, angle_deg) for the 12 hexagons under affine (s, sh) + global rot."""
    ca, sa = math.cos(rot), math.sin(rot)
    inner = []
    for q, r in _AXIAL:
        bx = (q + 0.5 * r) * _SQ3
        by = 1.5 * r
        # affine: x' = s*x + sh*y ; y' = s*y
        ax = s * bx + sh * by
        ay = s * by
        # global rotation
        x = ax * ca - ay * sa
        y = ax * sa + ay * ca
        inner.append([x, y, math.degrees(rot)])
    return inner


def _containing_side(inner):
    """Minimum side length of a regular hexagon (any center, any rotation) containing all inner hexes."""
    corners = []
    for x, y, ad in inner:
        corners.extend(_hex_corners(x, y, math.radians(ad)))
    min_side = None
    for k in range(6):
        rot = math.pi / 6 * k
        ca, sa = math.cos(rot), math.sin(rot)
        maxd = 0.0
        for x, y in corners:
            # support function of regular hexagon (circumradius R) rotated by rot
            a = -rot + math.pi / 3 * k
            d = abs(x * math.cos(a) + y * math.sin(a))
            if d > maxd:
                maxd = d
        R = maxd / _A60
        side = R
        if min_side is None or side < min_side:
            min_side = side
    return min_side


def _objective(params):
    s, sh, rot = params
    if s <= 0.05 or abs(sh) > 3.0 or abs(s) > 3.0:
        return 1e6
    inner = _build_inner(s, sh, rot)
    # penalty for overlaps
    corners = [_hex_corners(x, y, math.radians(ad)) for x, y, ad in inner]
    pen = 0.0
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if _hex_overlap(corners[i], corners[j]):
                pen += 1000.0
    if pen > 0:
        return _containing_side(inner) + pen
    return _containing_side(inner)


def _nelder_mead(f, x0, max_iter=400):
    """Simple Nelder-Mead; returns best x, best f."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] = pt[i] + 0.2 if pt[i] != 0 else 0.2
        simplex.append(pt)
    vals = [f(p) for p in simplex]
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        # centroid of all but worst
        cen = [sum(simplex[i][d] for i in range(n)) / n for d in range(n)]
        worst = simplex[n]
        # reflection
        alpha = 1.0
        xr = [cen[d] + alpha * (cen[d] - worst[d]) for d in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            gamma = 2.0
            xe = [cen[d] + gamma * (xr[d] - cen[d]) for d in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[n], vals[n] = xe, fe
            else:
                simplex[n], vals[n] = xr, fr
            continue
        if fr < vals[-1]:
            simplex[n], vals[n] = xr, fr
            continue
        # contraction
        beta = 0.5
        if fr < vals[-1]:
            xc = [cen[d] + beta * (worst[d] - cen[d]) for d in range(n)]
        else:
            xc = [cen[d] - beta * (cen[d] - worst[d]) for d in range(n)]
        fc = f(xc)
        if fc < vals[-1]:
            simplex[n], vals[n] = xc, fc
            continue
        # shrink
        for i in range(1, n + 1):
            simplex[i] = [simplex[0][d] + 0.5 * (simplex[i][d] - simplex[0][d]) for d in range(n)]
            vals[i] = f(simplex[i])
    order = sorted(range(n + 1), key=lambda i: vals[i])
    return simplex[order[0]], vals[order[0]]


def construct():
    best_inner, best_center, best_side, best_angle = None, (0.0, 0.0), None, 0.0

    starts = [(1.0, 0.0, 0.0), (0.95, 0.0, 0.0), (1.05, 0.0, 0.0),
              (1.0, 0.15, 0.0), (1.0, -0.15, 0.0), (0.95, 0.1, 0.2),
              (1.0, 0.0, 0.3), (1.05, -0.1, 0.1), (0.9, 0.2, 0.0),
              (1.0, 0.25, 0.15), (0.98, -0.2, 0.25), (1.02, 0.0, 0.15)]

    for s0, sh0, rot0 in starts:
        if time.time() > _DEADLINE:
            break
        x, val = _nelder_mead(_objective, [s0, sh0, rot0], max_iter=300)
        if val < 1e5:  # ignore heavily-penalized
            inner = _build_inner(*x)
            # verify no overlap
            corners = [_hex_corners(px, py, math.radians(ad)) for px, py, ad in inner]
            ok = True
            for i in range(len(corners)):
                for j in range(i + 1, len(corners)):
                    if _hex_overlap(corners[i], corners[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                side = _containing_side(inner)
                # center: bounding box midpoint
                xs = [c[0] for c in sum(corners, [])]
                ys = [c[1] for c in sum(corners, [])]
                cx = (min(xs) + max(xs)) / 2.0
                cy = (min(ys) + max(ys)) / 2.0
                # recenter inner so outer hexagon is centered at (cx,cy)
                inner = [[px - cx, py - cy, ad] for px, py, ad in inner]
                if best_side is None or side < best_side:
                    best_inner = inner
                    best_center = (0.0, 0.0)
                    best_side = side
                    best_angle = math.degrees(rot0) if False else 0.0

    if best_inner is None:
        # fallback: non-overlapping default
        inner = [[(q + 0.5 * r) * _SQ3, 1.5 * r, 0.0] for q, r in _AXIAL]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_angle
