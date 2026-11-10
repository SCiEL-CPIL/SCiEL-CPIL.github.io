# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 560.0  # safety margin below 600s wall clock


def _hex_verts(cx, cy, d, ang_rad):
    a = np.arange(6) * (np.pi / 3.0) + ang_rad
    return np.column_stack((cx + d * np.cos(a), cy + d * np.sin(a)))


def _min_outer_side(inner):
    """Smallest side length of a regular hexagon centered at origin (angle 0)
    that contains all inner unit hexagons (angle 0)."""
    pts = []
    for cx, cy, ang in inner:
        pts.append(_hex_verts(cx, cy, 1.0, ang))
    pts = np.concatenate(pts, axis=0)
    # outer hexagon centered at origin, side s, angle 0:
    # 6 vertices, 6 edges; each edge is a half-plane.
    s_min = 0.0
    for k in range(6):
        a1 = k * (np.pi / 3.0)
        a2 = (k + 1) * (np.pi / 3.0)
        p1 = np.array([np.cos(a1), np.sin(a1)])
        p2 = np.array([np.cos(a2), np.sin(a2)])
        e = p2 - p1
        n = np.array([-e[1], e[0]])  # outward normal (unit)
        n = n / np.linalg.norm(n)
        d_edge = n @ p1  # support function value for side 1
        max_dot = np.max(pts @ n)
        # need s * d_edge >= max_dot  ->  s >= max_dot / d_edge
        if d_edge > 1e-12:
            s_min = max(s_min, max_dot / d_edge)
    return s_min


def _validity(inner, outer, s):
    """Return True if the packing is valid per the task contract."""
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
        return False
    if s < np.sqrt(12.0) - 1e-9:
        return False
    ox, oy, oang = outer
    oang_r = np.deg2rad(oang)
    # outer vertices distinct
    ov = _hex_verts(ox, oy, s, oang_r)
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-8:
                return False
    # unit edge check after shifting by outer center
    for i in range(12):
        cx, cy, ang = inner[i]
        v = _hex_verts(cx - ox, cy - oy, 1.0, np.deg2rad(ang))
        for k in range(6):
            L = np.linalg.norm(v[(k + 1) % 6] - v[k])
            if abs(L - 1.0) > 1e-8:
                return False
    # inner-vertex-inside-outer check
    for i in range(12):
        cx, cy, ang = inner[i]
        v = _hex_verts(cx, cy, 1.0, np.deg2rad(ang))
        for p in v:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                if np.cross(p2 - p1, p - p1) < -1e-6:
                    return False
    # separation check via SAT on the 12 edge normals of each hexagon
    for i in range(12):
        for j in range(i + 1, 12):
            ci = inner[i]
            cj = inner[j]
            vi = _hex_verts(ci[0], ci[1], 1.0, np.deg2rad(ci[2]))
            vj = _hex_verts(cj[0], cj[1], 1.0, np.deg2rad(cj[2]))
            ok = False
            for vh in (vi, vj):
                for k in range(6):
                    e = vh[(k + 1) % 6] - vh[k]
                    n = np.array([-e[1], e[0]])
                    nl = np.linalg.norm(n)
                    if nl < 1e-12:
                        continue
                    n = n / nl
                    pi = vi @ n
                    pj = vj @ n
                    if (np.max(pi) - np.min(pj)) > 1e-6 or (np.max(pj) - np.min(pi)) > 1e-6:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                return False
    return True


def _pressure_energy(inner, s):
    """Total penetration energy between inner hexagons + confining energy
    from the outer hexagon of side s."""
    verts = [_hex_verts(c[0], c[1], 1.0, np.deg2rad(c[2])) for c in inner]
    e = 0.0
    # inner-inner penetration
    for i in range(12):
        for j in range(i + 1, 12):
            vi = verts[i]
            vj = verts[j]
            # use edges of both as candidate separating axes; penetration
            # depth along an axis = max overlap of intervals (positive if overlap)
            for vh in (vi, vj):
                for k in range(6):
                    ed = vh[(k + 1) % 6] - vh[k]
                    n = np.array([-ed[1], ed[0]])
                    nl = np.linalg.norm(n)
                    if nl < 1e-12:
                        continue
                    n = n / nl
                    pi = vi @ n
                    pj = vj @ n
                    pen = min(np.max(pi), np.max(pj)) - max(np.min(pi), np.min(pj))
                    if pen > 0:
                        e += pen * pen
    # confining: outer hexagon centered at origin, side s, angle 0
    for k in range(6):
        a1 = k * (np.pi / 3.0)
        a2 = (k + 1) * (np.pi / 3.0)
        p1 = np.array([np.cos(a1), np.sin(a1)])
        p2 = np.array([np.cos(a2), np.sin(a2)])
        ed = p2 - p1
        n = np.array([-ed[1], ed[0]])
        n = n / np.linalg.norm(n)
        d_edge = n @ p1
        for i in range(12):
            max_dot = np.max(verts[i] @ n)
            pen = (max_dot - s * d_edge) / max(d_edge, 1e-9)
            if pen > 0:
                e += pen * pen
    return e


def _random_init(rng):
    # Start from a compact hexagonal cluster arrangement, then jitter.
    base = [
        (0.0, 0.0),
        (2.5, 0.0), (-2.5, 0.0),
        (1.25, 2.1650635), (-1.25, 2.1650635),
        (1.25, -2.1650635), (-1.25, -2.1650635),
        (3.75, 2.1650635), (-3.75, 2.1650635),
        (3.75, -2.1650635), (-3.75, -2.1650635),
        (0.0, -4.330127),
    ]
    inner = np.zeros((12, 3))
    for i, (x, y) in enumerate(base):
        inner[i, 0] = x
        inner[i, 1] = y
        inner[i, 2] = 0.0
    inner[:, 0] += rng.normal(0, 0.5, 12)
    inner[:, 1] += rng.normal(0, 0.5, 12)
    inner[:, 2] += rng.normal(0, 15, 12)
    return inner


def hexagon_packing_12():
    rng = np.random.default_rng(0)
    best_inner = None
    best_s = np.inf
    best_val = -np.inf

    def record(inner):
        nonlocal best_inner, best_s, best_val
        s = _min_outer_side(inner)
        if not _validity(inner, np.array([0.0, 0.0, 0.0]), s):
            return
        val = 1.0 / s
        if val > best_val:
            best_val = val
            best_s = s
            best_inner = inner.copy()

    # Try multiple restarts with pressure-based simulated annealing.
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 30:
            break
        inner = _random_init(rng)
        T = 0.05
        step = 0.3
        best_local = inner.copy()
        best_local_s = _min_outer_side(best_local)
        best_local_val = -np.inf
        if _validity(best_local, np.array([0.0, 0.0, 0.0]), best_local_s):
            best_local_val = 1.0 / best_local_s
        iters = 4000
        for it in range(iters):
            if time.time() > _DEADLINE - 30:
                break
            # propose perturbation
            idx = rng.integers(0, 12)
            d = np.zeros(3)
            d[0] = rng.normal(0, step)
            d[1] = rng.normal(0, step)
            d[2] = rng.normal(0, step * 0.5)
            cand = inner.copy()
            cand[idx] += d
            # energy difference (use pressure energy as proxy)
            s_now = _min_outer_side(inner)
            s_cand = _min_outer_side(cand)
            e_now = _pressure_energy(inner, s_now)
            e_cand = _pressure_energy(cand, s_cand)
            de = e_cand - e_now
            if de < 0 or rng.random() < np.exp(-de / max(T, 1e-9)):
                inner = cand
                # track local best by validity + value
                if _validity(inner, np.array([0.0, 0.0, 0.0]), s_cand):
                    v = 1.0 / s_cand
                    if v > best_local_val:
                        best_local_val = v
                        best_local_s = s_cand
                        best_local = inner.copy()
            # adaptive step / temperature
            if it % 200 == 199:
                T *= 0.85
                step *= 0.92
        record(best_local)

    # Final polish on the best found
    if best_inner is not None:
        inner = best_inner.copy()
        step = 0.02
        for it in range(1500):
            if time.time() > _DEADLINE - 10:
                break
            idx = rng.integers(0, 12)
            d = np.zeros(3)
            d[0] = rng.normal(0, step)
            d[1] = rng.normal(0, step)
            d[2] = rng.normal(0, step * 0.3)
            cand = inner.copy()
            cand[idx] += d
            s_cand = _min_outer_side(cand)
            if _validity(cand, np.array([0.0, 0.0, 0.0]), s_cand):
                if (1.0 / s_cand) > best_val:
                    best_val = 1.0 / s_cand
                    best_s = s_cand
                    best_inner = cand.copy()
            step *= 0.999

    if best_inner is None:
        # Fallback to the seed arrangement (guaranteed valid)
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0],
            [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0],
        ], dtype=float)
        s = _min_outer_side(inner)
        return inner, np.array([0.0, 0.0, 0.0]), s

    return best_inner, np.array([0.0, 0.0, 0.0]), best_s


# EVOLVE-BLOCK-END
