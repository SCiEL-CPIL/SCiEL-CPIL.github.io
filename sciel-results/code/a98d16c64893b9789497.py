"""Packing 12 unit regular hexagons into a minimal regular hexagon.

Structural approach: the 12 hexagons form a rigid honeycomb cluster
(hexagonal-number arrangement: 1 center + 6 inner ring + 5 outer ring = 12).
The cluster's relative geometry is fixed by edge-contact constraints. The
only free parameter is the global rotation θ of the entire cluster. For each
θ, the enclosing regular hexagon's optimal orientation and side length are
computed analytically. We minimize the side length over θ.
"""
import math
import time

N = 12
SQR3 = math.sqrt(3.0)

# Deadline guard: return best found before the 150s wall-clock limit.
_DEADLINE = time.time() + 140.0  # 10s safety margin


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        th = math.radians(60.0 * k)
        x = math.cos(th)
        y = math.sin(th)
        verts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return verts


def _face_normals(angle_deg):
    """6 outward face normals of a regular hexagon rotated by angle_deg.
    For a hexagon with a vertex at angle 0, the face normals point at 30, 90, ..."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normals = []
    for k in range(6):
        th = math.radians(60.0 * k + 30.0)
        x = math.cos(th)
        y = math.sin(th)
        normals.append((x * ca - y * sa, x * sa + y * ca))
    return normals


def _min_enclosing_side(verts, outer_angle_deg):
    """Minimum side length of a regular hexagon (centered at origin, oriented at
    outer_angle_deg) that contains all points in verts.
    The hexagon's apothem is s*sqrt(3)/2, so s = max_k (n_k . p) / (sqrt(3)/2)."""
    normals = _face_normals(outer_angle_deg)
    max_proj = 0.0
    for (px, py) in verts:
        for (nx, ny) in normals:
            proj = px * nx + py * ny
            if proj > max_proj:
                max_proj = proj
    return max_proj / (SQR3 / 2.0)


def _base_positions():
    """Unrotated lattice positions of the 12 hexagon centers, centered at origin.
    Axial coordinates on a hex grid (pointy-top, size 1):
      x = sqrt(3) * (q + r/2),  y = 1.5 * r
    Cells: 1 center + 6 ring-1 + 5 ring-2 = 12.
    """
    cells = [
        (0, 0),
        # ring 1 (6)
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        # ring 2 (5 of 12) — chosen for compactness
        (2, 0), (-2, 0), (0, 2), (2, -2), (-2, 2),
    ]
    assert len(cells) == 12
    positions = []
    for (q, r) in cells:
        x = SQR3 * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
    # center at origin
    cx = sum(p[0] for p in positions) / 12.0
    cy = sum(p[1] for p in positions) / 12.0
    positions = [(p[0] - cx, p[1] - cy) for p in positions]
    return positions


_BASE_POS = _base_positions()


def _cluster_vertices(cluster_rotation_deg):
    """All 72 vertices of the 12 hexagons, each rotated by cluster_rotation_deg
    about the cluster centroid (origin), with centers at the rotated lattice
    positions. This is equivalent to rotating the entire unrotated cluster."""
    rot = math.radians(cluster_rotation_deg)
    cr, sr = math.cos(rot), math.sin(rot)
    verts = []
    for (bx, by) in _BASE_POS:
        # rotate center
        px = bx * cr - by * sr
        py = bx * sr + by * cr
        verts.extend(_hex_vertices(px, py, cluster_rotation_deg))
    return verts


def _evaluate(cluster_rotation_deg):
    """For a given cluster rotation, find the best enclosing hexagon orientation
    and return (side_length, outer_angle_deg)."""
    verts = _cluster_vertices(cluster_rotation_deg)
    best_side = float('inf')
    best_delta = 0.0
    # Scan outer orientation delta in [0, 60) relative to cluster frame.
    n_coarse = 120
    for i in range(n_coarse):
        delta = 60.0 * i / n_coarse
        outer_angle = cluster_rotation_deg + delta
        s = _min_enclosing_side(verts, outer_angle)
        if s < best_side:
            best_side = s
            best_delta = delta
    # Refine
    lo = best_delta - 60.0 / n_coarse
    hi = best_delta + 60.0 / n_coarse
    for _ in range(40):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        s1 = _min_enclosing_side(verts, cluster_rotation_deg + m1)
        s2 = _min_enclosing_side(verts, cluster_rotation_deg + m2)
        if s1 < s2:
            hi = m2
        else:
            lo = m1
    best_delta = (lo + hi) / 2.0
    best_side = _min_enclosing_side(verts, cluster_rotation_deg + best_delta)
    return best_side, (cluster_rotation_deg + best_delta) % 60.0


def construct():
    """Returns (inner, center, side, angle).
    inner: list of [x, y, angle_degrees] for the 12 unit hexagons.
    center: (x, y) of the outer hexagon.
    side: outer hexagon side length.
    angle: outer hexagon rotation in degrees.
    """
    t_start = time.time()
    best_side = float('inf')
    best_cluster_rot = 0.0
    best_outer_angle = 0.0

    # Coarse scan of cluster rotation over [0, 60) by 6-fold symmetry.
    n_coarse = 24
    for i in range(n_coarse):
        if time.time() > _DEADLINE:
            break
        cr = 60.0 * i / n_coarse
        s, oa = _evaluate(cr)
        if s < best_side:
            best_side = s
            best_cluster_rot = cr
            best_outer_angle = oa

    # Refine cluster rotation via ternary search.
    if time.time() < _DEADLINE:
        lo = best_cluster_rot - 60.0 / n_coarse
        hi = best_cluster_rot + 60.0 / n_coarse
        iters = 0
        max_iters = 80
        while (hi - lo) > 1e-5 and iters < max_iters:
            if time.time() > _DEADLINE:
                break
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            s1, oa1 = _evaluate(m1)
            s2, oa2 = _evaluate(m2)
            if s1 < s2:
                hi = m2
            else:
                lo = m1
            if s1 < best_side:
                best_side = s1
                best_cluster_rot = m1
                best_outer_angle = oa1
            if s2 < best_side:
                best_side = s2
                best_cluster_rot = m2
                best_outer_angle = oa2
            iters += 1
        # Final
        s, oa = _evaluate(best_cluster_rot)
        if s < best_side:
            best_side = s
            best_outer_angle = oa

    # Build inner list: each hexagon at rotated center, rotated by cluster_rotation.
    rot = math.radians(best_cluster_rot)
    cr, sr = math.cos(rot), math.sin(rot)
    inner = []
    for (bx, by) in _BASE_POS:
        px = bx * cr - by * sr
        py = bx * sr + by * cr
        inner.append([px, py, best_cluster_rot])

    center = (0.0, 0.0)
    side = best_side
    angle = best_outer_angle

    return inner, center, side, angle


if __name__ == "__main__":
    t0 = time.time()
    inner, center, side, angle = construct()
    print(f"side={side:.6f}, angle={angle:.4f}, time={time.time()-t0:.2f}s")
    print(f"num inner={len(inner)}")
    # Quick overlap sanity check: verify no vertex of one hex is strictly inside
    # another hexagon.
    def _point_in_hex(px, py, hx, hy, ha_deg, eps=1e-9):
        a = math.radians(ha_deg)
        ca, sa = math.cos(a), math.sin(a)
        # transform point into hex local frame
        dx = px - hx
        dy = py - hy
        lx = dx * ca + dy * sa
        ly = -dx * sa + dy * ca
        # hex with vertex at angle 0: |lx| <= 1 and |ly| <= sqrt(3)/2 and
        # |lx| + |ly|/sqrt(3) <= 1  ... use the 6 half-plane test
        for k in range(6):
            th = math.radians(60.0 * k + 30.0)
            nx, ny = math.cos(th), math.sin(th)
            if lx * nx + ly * ny > SQR3 / 2.0 + eps:
                return False
        return True
    ok = True
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            # check a sample of vertices of hex i against hex j
            for (vx, vy) in _hex_vertices(inner[i][0], inner[i][1], inner[i][2]):
                if _point_in_hex(vx, vy, inner[j][0], inner[j][1], inner[j][2]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    print(f"non-overlap check: {'PASS' if ok else 'FAIL'}")
    print("OK")
