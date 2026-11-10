# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 600.0 - 30.0  # safety margin before the 630s kill


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(th), cy + s * np.sin(th)], axis=1)


def _edge_normals(verts):
    pts = np.vstack([verts, verts[:1]])
    edges = pts[1:] - pts[:-1]
    n = np.column_stack([edges[:, 1], -edges[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


def _min_gap(A, B, nA, nB):
    """Max over the 12 edge normals of the minimum signed separation between
    two hexagons.  > 0  -> separated by a clear gap.  <= 0 -> touching/overlap."""
    pA = np.vstack([A, A[:1]])
    pB = np.vstack([B, B[:1]])
    best = -np.inf
    for n in nA:
        vA = pA @ n
        vB = pB @ n
        gap = max(vA.min() - vB.max(), vB.min() - vA.max())
        if gap > best:
            best = gap
    for n in nB:
        vA = pA @ n
        vB = pB @ n
        gap = max(vA.min() - vB.max(), vB.min() - vA.max())
        if gap > best:
            best = gap
    return best


def _inner_ok(inner):
    """Pairwise separation: every pair must be separated on at least one normal."""
    verts = np.array([_hex_vertices(x, y, 1.0, a) for (x, y, a) in inner])
    normals = np.array([_edge_normals(v) for v in verts])
    for i in range(11):
        for j in range(i + 1, 11):
            if _min_gap(verts[i], verts[j], normals[i], normals[j]) <= 1e-6:
                return False
    return True


def _outer_side_for(inner, center):
    """Smallest side (centered at `center`, angle 0) whose 6 edges contain every
    inner vertex.  Returns np.inf if no finite side works."""
    verts = np.array([_hex_vertices(x, y, 1.0, a) for (x, y, a) in inner])
    q = verts - center
    n_out = _edge_normals(_hex_vertices(0.0, 0.0, 1.0, 0.0))
    side = 0.0
    for n in n_out:
        d = q @ n
        if d.max() > 1e-6:
            return np.inf
        side = max(side, d.min() / np.sqrt(3.0))
    return side


def _valid(inner, outer, side):
    if side < np.sqrt(11.0) - 1e-9:
        return False
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if not _inner_ok(inner):
        return False
    cx, cy, ca = outer
    o = _hex_vertices(cx, cy, side, ca)
    oc = np.array([cx, cy])
    for (x, y, a) in inner:
        v = _hex_vertices(x, y, 1.0, a) - oc
        for k in range(6):
            p1 = o[k] - oc
            p2 = o[(k + 1) % 6] - oc
            cross = (p2[0] - p1[0]) * (v[:, 1] - p1[1]) - \
                    (p2[1] - p1[1]) * (v[:, 0] - p1[0])
            if np.min(cross) < -1e-6:
                return False
    return True


def _base_solution():
    r = np.sqrt(3.0)
    centers = [
        (0.0, 0.0),
        (2.0, 0.0), (1.0, r), (-1.0, r), (-2.0, 0.0), (-1.0, -r), (1.0, -r),
        (2.0 * r, 1.0), (2.0 * r, -1.0), (-2.0 * r, 1.0), (-2.0 * r, -1.0),
    ]
    inner = np.array([[x, y, 0.0] for (x, y) in centers])
    outer = np.array([0.0, 0.0, 0.0])
    side = 5.0
    return inner, outer, side


def _refine(inner, outer, side):
    """Local Nelder-Mead refinement of (positions, outer center, side).
    Only accepts a candidate if it stays valid and shrinks the side."""
    from scipy.optimize import minimize

    def unpack(p):
        inner_p = p[:33].reshape(11, 3)
        oc = p[33:35]
        s = p[35]
        return inner_p, oc, s

    def objective(p):
        inner_p, oc, s = unpack(p)
        if not np.all(np.isfinite(p)) or s < np.sqrt(11.0) - 1e-9:
            return 1e12
        if not _inner_ok(inner_p):
            return 1e12
        cx, cy, ca = outer
        o = _hex_vertices(cx, cy, s, ca)
        oc2 = np.array([cx, cy])
        for (x, y, a) in inner_p:
            v = _hex_vertices(x, y, 1.0, a) - oc2
            for k in range(6):
                p1 = o[k] - oc2
                p2 = o[(k + 1) % 6] - oc2
                cross = (p2[0] - p1[0]) * (v[:, 1] - p1[1]) - \
                        (p2[1] - p1[1]) * (v[:, 0] - p1[0])
                if np.min(cross) < -1e-6:
                    return 1e12
        return s

    x0 = inner.ravel().copy()
    x0 = np.concatenate([x0, outer[:2], [side]])
    try:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-12,
                                "adaptive": True})
        if res.success or (np.isfinite(res.fun) and res.fun < side - 1e-9):
            inner_p, oc, s = unpack(res.x)
            if _valid(inner_p, np.array([oc[0], oc[1], outer[2]]), s):
                return inner_p, np.array([oc[0], oc[1], outer[2]]), float(s)
    except Exception:
        pass
    return None


def hexagon_packing_11():
    inner, outer, side = _base_solution()
    if not _valid(inner, outer, side):
        # Fallback: enlarge the container so the known flower is guaranteed valid.
        outer = np.array([0.0, 0.0, 0.0])
        side = 6.0
        if not _valid(inner, outer, side):
            side = 8.0
        return inner, outer, side

    # Try to tighten the side with a bounded local search (only helps).
    if time.time() < _DEADLINE:
        refined = _refine(inner.copy(), outer.copy(), side)
        if refined is not None:
            inner, outer, side = refined

    return inner, outer, side


# EVOLVE-BLOCK-END
