# EVOLVE-BLOCK-START
import time
import numpy as np

_TIME_START = time.time()
_DEADLINE = 630.0
_MARGIN = 30.0  # seconds of safety margin


def _hexagon_vertices(cx, cy, d, a_deg):
    """Vertices of a regular hexagon: center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + d * np.cos(th), cy + d * np.sin(th)))


def _rot(x, y, ang):
    c, s = np.cos(ang), np.sin(ang)
    return x * c - y * s, x * s + y * c


def _build_cluster():
    """Rigid honeycomb cluster of 12 unit hexagons (3-4-3-2 rows)."""
    dx = 2.5
    dy = np.sqrt(3) * 1.5
    rows = {0: 3, 1: 4, 2: 3, 3: 2}
    centers = []
    for r, cnt in rows.items():
        y = dy * (r - 1.5)
        for i in range(cnt):
            x = dx * (i - (cnt - 1) / 2.0)
            centers.append((x, y))
    return np.array(centers, dtype=float)


def _min_s(vertices, ang_o, cx, cy):
    """Minimal side length s so that all vertices fit inside the outer hexagon
    (center (cx,cy), angle ang_o, side s)."""
    a = np.deg2rad(ang_o)
    th = a + np.deg2rad(60.0) * np.arange(6)
    v = vertices - np.array([cx, cy])
    s = 0.0
    for i in range(6):
        n = np.array([-np.sin(th[i]), np.cos(th[i])])
        p1 = s0 = None
        # support of outer hexagon along outward normal n:
        # outer edge i has outward normal n_i; all points must satisfy dot(p, n_i) <= s
        proj = v @ n
        s = max(s, proj.max())
    return s


def _evaluate(ang_c, ang_o, cx, cy, local_pts):
    """Return enclosing side length s for cluster rotated by ang_c,
    outer rotated by ang_o centered at (cx,cy)."""
    c, sgn = np.cos(ang_c), np.sin(ang_c)
    rx = local_pts[:, 0] * c - local_pts[:, 1] * sgn
    ry = local_pts[:, 0] * sgn + local_pts[:, 1] * c
    verts = []
    for i in range(len(local_pts)):
        verts.append(_hexagon_vertices(rx[i], ry[i], 1.0, 0.0))
    allv = np.vstack(verts)
    return _min_s(allv, ang_o, cx, cy)


def hexagon_packing_12():
    cluster = _build_cluster()  # (12,2)
    # precompute all 12*6 vertices in cluster frame (angle 0)
    local_pts = []
    for i in range(len(cluster)):
        hv = _hexagon_vertices(cluster[i, 0], cluster[i, 1], 1.0, 0.0)
        local_pts.append(hv)
    local_pts = np.vstack(local_pts)  # (72,2)

    best = None
    best_s = float("inf")

    # Coarse grid over cluster rotation (0..60deg, periodic), outer rotation (0..60),
    # and center offset.
    def try_cfg(ang_c_deg, ang_o_deg, cx, cy):
        nonlocal best, best_s
        if time.time() > _DEADLINE - _MARGIN:
            return False
        s = _evaluate(np.deg2rad(ang_c_deg), np.deg2rad(ang_o_deg), cx, cy, local_pts)
        if s < best_s:
            best_s = s
            best = (ang_c_deg, ang_o_deg, cx, cy, s)
        return True

    # The cluster is symmetric; outer hexagon symmetric under 60deg.
    # Try a range of cluster rotations and outer rotations, with center at origin
    # (by symmetry the optimal center aligns with cluster centroid; but allow search).
    centroid = cluster.mean(axis=0)

    # Grid search
    ang_c_vals = np.arange(0, 60, 5)
    ang_o_vals = np.arange(0, 60, 5)
    # center offset candidates around centroid (in cluster frame) -> but we rotate cluster,
    # so keep center at origin and let rotation do the work. Try small offsets too.
    offset_vals = [0.0]
    for ang_c in ang_c_vals:
        for ang_o in ang_o_vals:
            for off in offset_vals:
                cx = centroid[0] * np.cos(np.deg2rad(ang_c)) - centroid[1] * np.sin(np.deg2rad(ang_c)) + off
                cy = centroid[0] * np.sin(np.deg2rad(ang_c)) + centroid[1] * np.cos(np.deg2rad(ang_c)) + off
                if not try_cfg(ang_c, ang_o, cx, cy):
                    break

    # Refine with local coordinate descent around the best.
    if best is not None:
        ang_c, ang_o, cx, cy, s = best
        step_c = 1.0
        step_o = 1.0
        step_xy = 0.1
        for _ in range(60):
            if time.time() > _DEADLINE - _MARGIN:
                break
            improved = False
            for da in (-step_c, step_c):
                if try_cfg(ang_c + da, ang_o, cx, cy):
                    ang_c, ang_o, cx, cy, s = best
                    improved = True
            for da in (-step_o, step_o):
                if try_cfg(ang_c, ang_o + da, cx, cy):
                    ang_c, ang_o, cx, cy, s = best
                    improved = True
            for dx in (-step_xy, step_xy):
                if try_cfg(ang_c, ang_o, cx + dx, cy):
                    ang_c, ang_o, cx, cy, s = best
                    improved = True
            for dy in (-step_xy, step_xy):
                if try_cfg(ang_c, ang_o, cx, cy + dy):
                    ang_c, ang_o, cx, cy, s = best
                    improved = True
            if not improved:
                step_c *= 0.5
                step_o *= 0.5
                step_xy *= 0.5
                if step_c < 1e-4 and step_xy < 1e-5:
                    break

    # Finalize best configuration
    ang_c, ang_o, cx, cy, s = best
    c, sgn = np.cos(np.deg2rad(ang_c)), np.sin(np.deg2rad(ang_c))
    inner = []
    for i in range(len(cluster)):
        x0, y0 = cluster[i, 0], cluster[i, 1]
        rx = x0 * c - y0 * sgn
        ry = x0 * sgn + y0 * c
        inner.append([rx, ry, 0.0])
    inner = np.array(inner, dtype=float)
    outer = np.array([cx, cy, ang_o], dtype=float)
    s = float(s)
    return inner, outer, s


# EVOLVE-BLOCK-END
