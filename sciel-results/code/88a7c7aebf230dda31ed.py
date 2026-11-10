# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 630.0 - 15.0  # safety margin before the 630s kill

def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack((cx + d * np.cos(ang), cy + d * np.sin(ang)))

def _min_containing_hex_side(pts, outer_center=None):
    """Smallest side length of an axis-aligned (angle 0) regular hexagon,
    optionally with fixed center, that contains all points."""
    if outer_center is None:
        xc = 0.5 * (pts[:, 0].max() - pts[:, 0].min())
        yc = 0.5 * (pts[:, 1].max() - pts[:, 1].min())
    else:
        xc, yc = outer_center
    dx = pts[:, 0] - xc
    dy = pts[:, 1] - yc
    # vertex directions for angle 0: 0,60,120,180,240,300
    c = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    s = np.array([0.0, np.sqrt(3) / 2, np.sqrt(3) / 2, 0.0, -np.sqrt(3) / 2, -np.sqrt(3) / 2])
    proj = dx @ c + dy @ s  # (6, N)
    mx = proj.max(axis=1)
    mn = proj.min(axis=1)
    return np.max(np.maximum(mx, -mn))

def _check_separation(inner):
    """Return True if every pair of inner unit hexagons is separated
    (SAT over the 12 edge normals)."""
    n = len(inner)
    verts = []
    for (cx, cy, ang) in inner:
        verts.append(_hex_vertices(cx, cy, 1.0, ang))
    normals = []
    for v in verts:
        e = np.diff(np.vstack([v, v[:1]]), axis=0)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-30)
        normals.append(np.column_stack((-e[:, 1], e[:, 0])))
    for i in range(n):
        for j in range(i + 1, n):
            sep = False
            for nm in normals[i]:
                pi = verts[i] @ nm
                pj = verts[j] @ nm
                if max(pi.min(), pj.min()) < min(pi.max(), pj.max()) - 1e-9:
                    sep = True
                    break
            if not sep:
                for nm in normals[j]:
                    pi = verts[i] @ nm
                    pj = verts[j] @ nm
                    if max(pi.min(), pj.min()) < min(pi.max(), pj.max()) - 1e-9:
                        sep = True
                        break
            if not sep:
                return False
    return True

def _build(inner, outer_center, outer_angle):
    pts = np.vstack([_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in inner])
    if outer_center is None:
        xc = 0.5 * (pts[:, 0].max() - pts[:, 0].min())
        yc = 0.5 * (pts[:, 1].max() - pts[:, 1].min())
    else:
        xc, yc = outer_center
    dx = pts[:, 0] - xc
    dy = pts[:, 1] - yc
    a = np.deg2rad(outer_angle)
    c = np.cos(a + np.deg2rad(np.arange(6) * 60.0))
    s = np.sin(a + np.deg2rad(np.arange(6) * 60.0))
    proj = dx @ c + dy @ s
    mx = proj.max(axis=1)
    mn = proj.min(axis=1)
    return float(np.max(np.maximum(mx, -mn)))

def _objective(x, ring, base_angle):
    # x: [cx0..cx4, cy0..cy4, outer_angle]
    ring_xy = x[:10].reshape(5, 2)
    oa = x[10]
    inner = []
    for i in range(7):
        inner.append((0.0, 0.0, 0.0))
    for i in range(5):
        inner.append((ring_xy[i, 0], ring_xy[i, 1], base_angle))
    if not _check_separation(inner):
        return 1e6
    return _build(inner, None, oa)

def _seed_positions():
    # 7-hex cluster centers (side 1) + 5 ring hexagons at 5 of 6 outer-ring slots
    base = [
        (0.0, 0.0),
        (2.5, 0.0), (1.25, 2.1650635094610964), (-1.25, 2.1650635094610964),
        (-2.5, 0.0), (-1.25, -2.1650635094610964), (1.25, -2.1650635094610964),
    ]
    ring6 = [
        (5.0, 0.0), (3.75, 4.330127018922193), (0.0, 4.330127018922193),
        (-3.75, 4.330127018922193), (-5.0, 0.0), (-3.75, -4.330127018922193),
        (0.0, -4.330127018922193), (3.75, -4.330127018922193),
    ]
    # choose 5 of the 8 outer-ring slots (a valid non-overlapping 5-subset)
    chosen = [0, 1, 2, 3, 4]  # 5 consecutive slots -> valid
    ring = [ring6[k] for k in chosen]
    return base, ring

def _nelder_mead(f, x0, maxiter=800, tol=1e-9):
    from scipy.optimize import minimize
    res = minimize(f, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": tol, "fatol": tol})
    return res.x, res.fun

def hexagon_packing_12():
    base, ring = _seed_positions()
    base_angle = 0.0
    best = None
    best_val = np.inf

    def consider(inner, outer_center, outer_angle, s):
        nonlocal best, best_val
        if s < best_val - 1e-12:
            best_val = s
            best = (inner, outer_center, outer_angle, s)

    # 1) analytic baseline: fixed cluster, tight outer hex (angle 0, centered)
    inner0 = [(0.0, 0.0, 0.0)] * 7 + [(rx, ry, base_angle) for (rx, ry) in ring]
    s0 = _build(inner0, None, 0.0)
    consider(inner0, (0.0, 0.0), 0.0, s0)

    # 2) Nelder-Mead over ring positions + outer angle
    x0 = np.array([c for r in ring for c in r] + [0.0])
    try:
        xopt, fopt = _nelder_mead(lambda x: _objective(x, ring, base_angle), x0, maxiter=600)
        if fopt < 1e5:
            ring_xy = xopt[:10].reshape(5, 2)
            oa = xopt[10]
            inner = [(0.0, 0.0, 0.0)] * 7 + [(ring_xy[i, 0], ring_xy[i, 1], base_angle) for i in range(5)]
            s = _build(inner, None, oa)
            consider(inner, (0.0, 0.0), oa, s)
    except Exception:
        pass

    # 3) refine outer center for the best inner set (tighten)
    if best is not None:
        inner_b, _, oa_b, s_b = best
        pts = np.vstack([_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in inner_b])
        # try a small grid of centers around the bounding-box center
        cx0 = 0.5 * (pts[:, 0].max() - pts[:, 0].min())
        cy0 = 0.5 * (pts[:, 1].max() - pts[:, 1].min())
        for dcx in np.linspace(-0.5, 0.5, 9):
            for dcy in np.linspace(-0.5, 0.5, 9):
                s = _build(inner_b, (cx0 + dcx, cy0 + dcy), oa_b)
                consider(inner_b, (cx0 + dcx, cy0 + dcy), oa_b, s)

    if best is None:
        # fallback: seed-style valid packing
        inner0 = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0], [-3.75, -2.17, 0], [3.75, -2.17, 0],
            [0, -4, 0],
        ], dtype=float)
        outer0 = np.array([0.0, 0.0, 0.0])
        return inner0, outer0, 8.0

    inner_b, oc_b, oa_b, s_b = best
    inner_arr = np.array(inner_b, dtype=float)
    outer_arr = np.array([oc_b[0], oc_b[1], oa_b], dtype=float)
    return inner_arr, outer_arr, float(s_b)

# EVOLVE-BLOCK-END
