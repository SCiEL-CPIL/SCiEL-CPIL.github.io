"""Pack 12 unit regular hexagons inside a regular hexagon minimizing outer side length."""
import time
import math
import random

import numpy as np

N = 12
DEADLINE = 140.0  # safety margin before 150s kill
RNG_SEED = 12345


def _hex_vertices(side, cx, cy, angle_deg):
    """Vertices of a regular hexagon (flat-top at angle 0) with given side, center, rotation."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + math.radians(60.0 * k)
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts


def _separation_hex_hex(v1, v2, eps=1e-9):
    """
    Minimum distance to make two convex hexagons (lists of (x,y)) non-overlapping.
    Returns (gap, dx, dy) where gap>=0 if separated (distance between them),
    gap<0 if overlapping (penetration depth); (dx,dy) is direction to push v2 away.
    Uses separating axis theorem on convex polygons.
    """
    # Edge normals of both polygons
    axes = []
    for poly in (v1, v2):
        for i in range(6):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 6]
            ex, ey = x2 - x1, y2 - y1
            L = math.hypot(ex, ey)
            if L < 1e-12:
                continue
            # outward normal (for CCW polygon, normal is (ey, -ex) normalized)
            nx, ny = ey / L, -ex / L
            axes.append((nx, ny))
    best_axis = None
    best_gap = float('inf')
    best_pen = 0.0  # max penetration (we want the axis with least overlap)
    found_overlap = False
    for (nx, ny) in axes:
        # project both polygons
        min1 = min(p[0]*nx + p[1]*ny for p in v1)
        max1 = max(p[0]*nx + p[1]*ny for p in v1)
        min2 = min(p[0]*nx + p[1]*ny for p in v2)
        max2 = max(p[0]*nx + p[1]*ny for p in v2)
        # gap: if separated
        if max1 + eps <= min2 or max2 + eps <= min1:
            # separating axis found -> no overlap
            gap = max(min2 - max1, min1 - max2)
            if gap < best_gap:
                best_gap = gap
                best_axis = (nx, ny)
            return (gap, nx, ny)  # separated
        else:
            # overlap on this axis; penetration = overlap amount
            pen = min(max1, max2) - max(min1, min2)
            if pen > best_pen:
                best_pen = pen
                best_axis = (nx, ny)
    # No separating axis -> overlapping; return deepest penetration axis
    return (-best_pen, best_axis[0], best_axis[1])


def _separation_hex_hex_fast(v1, v2, centers, eps=1e-9):
    """
    Faster SAT: only check axes from v1's edges (6) plus v2's edges (6) = 12 axes.
    Returns (gap, dx, dy). gap>0 separated, gap<0 overlap depth.
    """
    min_pen = float('inf')
    best_ax = (1.0, 0.0)
    # Precompute projections once per axis is expensive; do per axis.
    for poly in (v1, v2):
        for i in range(6):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 6]
            ex, ey = x2 - x1, y2 - y1
            L = math.hypot(ex, ey)
            if L < 1e-12:
                continue
            nx, ny = ey / L, -ex / L
            # projections
            p1 = [p[0]*nx + p[1]*ny for p in v1]
            p2 = [p[0]*nx + p[1]*ny for p in v2]
            min1, max1 = min(p1), max(p1)
            min2, max2 = min(p2), max(p2)
            if max1 + eps <= min2 or max2 + eps <= min1:
                # separated on this axis
                gap = max(min2 - max1, min1 - max2)
                return (gap, nx, ny)
            else:
                pen = min(max1, max2) - max(min1, min2)
                if pen < min_pen:
                    min_pen = pen
                    best_ax = (nx, ny)
    # overlapping: return penetration (negative gap)
    return (-min_pen, best_ax[0], best_ax[1])


def _outer_contains(inner_verts, outer_verts):
    """Check if all inner verts are inside (or on) the outer hexagon."""
    # Outer hexagon is convex; use edge half-plane test.
    # For each edge of outer, check all inner points are on the interior side.
    # Determine interior side via center of outer.
    cx = sum(v[0] for v in outer_verts) / 6.0
    cy = sum(v[1] for v in outer_verts) / 6.0
    for i in range(6):
        x1, y1 = outer_verts[i]
        x2, y2 = outer_verts[(i + 1) % 6]
        ex, ey = x2 - x1, y2 - y1
        L = math.hypot(ex, ey)
        if L < 1e-12:
            continue
        nx, ny = ey / L, -ex / L  # normal
        # interior side: center should be on same side as interior
        # sign of (center - p1) dot normal
        s_center = (cx - x1) * nx + (cy - y1) * ny
        for (px, py) in inner_verts:
            s = (px - x1) * nx + (py - y1) * ny
            if s_center > 0:
                if s < -1e-7:
                    return False
            else:
                if s > 1e-7:
                    return False
    return True


def _violation(inner, outer_center, outer_side, outer_angle):
    """
    Return total violation: sum of overlaps between inner hexagons
    plus how far inner hexagons extend outside outer hexagon.
    Lower is better; 0 means feasible.
    """
    outer_verts = _hex_vertices(outer_side, outer_center[0], outer_center[1], outer_angle)
    total = 0.0
    inner_verts_list = []
    for (x, y, ang) in inner:
        inner_verts_list.append(_hex_vertices(1.0, x, y, ang))
    # pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            gap, dx, dy = _separation_hex_hex_fast(inner_verts_list[i], inner_verts_list[j])
            if gap < 0:
                total += (-gap) ** 2
    # containment: each inner vertex must be inside outer
    for iv in inner_verts_list:
        cx = sum(v[0] for v in outer_verts) / 6.0
        cy = sum(v[1] for v in outer_verts) / 6.0
        for i in range(6):
            x1, y1 = outer_verts[i]
            x2, y2 = outer_verts[(i + 1) % 6]
            ex, ey = x2 - x1, y2 - y1
            L = math.hypot(ex, ey)
            if L < 1e-12:
                continue
            nx, ny = ey / L, -ex / L
            s_center = (cx - x1) * nx + (cy - y1) * ny
            for (px, py) in iv:
                s = (px - x1) * nx + (py - y1) * ny
                if s_center > 0:
                    if s < 0:
                        total += (-s) ** 2
                else:
                    if s > 0:
                        total += (s) ** 2
    return total


def _initial_layout():
    """A reasonable starting layout: 12 hexagons in a honeycomb cluster."""
    # Honeycomb offsets for a cluster of 12 hexagons.
    # Use axial coordinates.
    axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1),
    ]
    # Convert axial to pixel for pointy-top hexagons with side 1.
    # For pointy-top: x = sqrt(3)*s*(q + r/2), y = 1.5*s*r
    # We'll use flat-top convention to match _hex_vertices (angle 0 = flat-top).
    # Flat-top: x = 1.5*s*q, y = sqrt(3)*s*(r + q/2)
    s = 1.0
    pos = []
    for (q, r) in axial:
        x = 1.5 * s * q
        y = math.sqrt(3.0) * s * (r + q / 2.0)
        pos.append([x, y, 0.0])
    return pos


def _anneal(initial, outer_center, outer_side, outer_angle, start_temp, end_temp, iters, deadline):
    """Simulated annealing on inner positions and angles, plus outer side."""
    best_inner = [row[:] for row in initial]
    best_outer = (outer_center, outer_side, outer_angle)
    best_viol = _violation(best_inner, outer_center, outer_side, outer_angle)
    cur_inner = [row[:] for row in initial]
    cur_side = outer_side
    cur_viol = best_viol
    temp = start_temp
    for it in range(iters):
        if time.time() > deadline:
            break
        # Pick a random inner hexagon to perturb
        idx = random.randrange(N)
        dx = random.gauss(0, 0.3)
        dy = random.gauss(0, 0.3)
        dangle = random.gauss(0, 15.0)
        # Occasionally perturb outer side
        if random.random() < 0.1:
            ds = random.gauss(0, 0.2)
        else:
            ds = 0.0
        new_inner = [row[:] for row in cur_inner]
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += dangle
        new_side = cur_side + ds
        if new_side < 2.0:
            new_side = 2.0
        new_viol = _violation(new_inner, outer_center, new_side, outer_angle)
        delta = new_viol - cur_viol
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
            cur_inner = new_inner
            cur_side = new_side
            cur_viol = new_viol
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_inner = [row[:] for row in cur_inner]
                best_outer = (outer_center, cur_side, outer_angle)
        temp *= (end_temp / start_temp) ** (1.0 / max(iters, 1))
    return best_inner, best_outer, best_viol


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    # Start with a honeycomb layout; estimate a tight outer side.
    initial = _initial_layout()
    # Compute bounding extent to estimate outer side.
    xs = [p[0] for p in initial]
    ys = [p[1] for p in initial]
    # Add radius of each hex (circumradius = 1 for side 1)
    max_x = max(xs) + 1.5
    min_x = min(xs) - 1.5
    max_y = max(ys) + math.sqrt(3)
    min_y = min(ys) - math.sqrt(3)
    # For a flat-top hexagon of side S, width = 2*S, height = sqrt(3)*S
    # We need to cover the bounding box; side must be at least max(width, height/sqrt(3))
    width = max_x - min_x
    height = max_y - min_y
    side_est = max(width / 2.0, height / math.sqrt(3.0)) + 0.5
    outer_center = ((max_x + min_x) / 2.0, (max_y + min_y) / 2.0)
    outer_angle = 0.0

    # First, do a quick local optimization to get a feasible-ish starting point.
    best_inner, best_outer, best_viol = _anneal(
        initial, outer_center, side_est, outer_angle,
        start_temp=1.0, end_temp=0.01, iters=2000, deadline=deadline
    )

    # Now try to reduce outer side by re-centering and tightening.
    # Repeatedly: recompute optimal outer center/side for current inner, then anneal.
    for round in range(6):
        if time.time() > deadline:
            break
        # Estimate minimal enclosing hexagon side for current inner layout.
        # Use a simple approach: try several outer angles and centers, pick best.
        inner = best_inner
        # Compute center of mass of inner hexagons
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        best_side = float('inf')
        best_ox = cx
        best_oy = cy
        best_oa = 0.0
        for oa in range(0, 60, 15):
            # For this angle, find minimal side by scaling from center.
            # Compute for each inner vertex its distance from center in hex metric.
            # Approximate: use max over vertices of support function.
            # Simpler: binary search on side.
            lo = 1.0
            hi = side_est + 5.0
            for _ in range(25):
                mid = (lo + hi) / 2.0
                ov = _hex_vertices(mid, cx, cy, oa)
                # Check containment
                ok = True
                for (x, y, ang) in inner:
                    iv = _hex_vertices(1.0, x, y, ang)
                    for (px, py) in iv:
                        # point in hexagon test
                        inside = True
                        for i in range(6):
                            x1, y1 = ov[i]
                            x2, y2 = ov[(i + 1) % 6]
                            ex, ey = x2 - x1, y2 - y1
                            L = math.hypot(ex, ey)
                            if L < 1e-12:
                                continue
                            nx, ny = ey / L, -ex / L
                            s_c = (cx - x1) * nx + (cy - y1) * ny
                            s = (px - x1) * nx + (py - y1) * ny
                            if s_c > 0:
                                if s < -1e-7:
                                    inside = False
                                    break
                            else:
                                if s > 1e-7:
                                    inside = False
                                    break
                        if not inside:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    hi = mid
                else:
                    lo = mid
            if hi < best_side:
                best_side = hi
                best_oa = oa
        # Now anneal with this tighter outer, allowing outer side to grow if needed.
        new_inner, new_outer, new_viol = _anneal(
            best_inner, (best_ox, best_oy), best_side, best_oa,
            start_temp=0.5, end_temp=0.01, iters=3000, deadline=deadline
        )
        if new_viol < best_viol or (new_viol < 1e-4 and new_outer[1] < best_outer[1]):
            best_inner = new_inner
            best_outer = new_outer
            best_viol = new_viol
        if time.time() > deadline:
            break

    # Final: return the best found. If violation is not zero, we still return best effort.
    # But we should try to ensure feasibility by slightly inflating outer side if needed.
    inner = best_inner
    ocx, ocy = best_outer[0]
    oside = best_outer[1]
    oangle = best_outer[2]
    # Check feasibility; if not, inflate outer side a bit.
    for _ in range(50):
        if time.time() > deadline:
            break
        v = _violation(inner, (ocx, ocy), oside, oangle)
        if v < 1e-6:
            break
        oside += 0.05
    return inner, (ocx, ocy), oside, oangle


if __name__ == "__main__":
    t = time.time()
    inner, center, side, angle = construct()
    print("side:", side, "time:", time.time() - t)
