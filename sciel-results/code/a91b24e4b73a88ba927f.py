# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START = time.time()
_DEADLINE = 630.0 - 6.0  # safety margin before the 630s kill
_EPS = 1e-6


def _hex_vertices(cx, cy, s, a_deg):
    a = math.radians(a_deg)
    return np.array([
        [cx + s * math.cos(a + math.radians(60 * i)),
         cy + s * math.sin(a + math.radians(60 * i))]
        for i in range(6)
    ])


def _inner_vertices(positions, angles_deg):
    n = len(positions)
    verts = np.empty((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
    return verts


def _is_valid(positions, angles_deg, outer_center, outer_angle_deg, side):
    # finite checks
    if not (np.all(np.isfinite(positions)) and np.isfinite(outer_center).all()
            and np.isfinite(outer_angle_deg) and np.isfinite(side)):
        return False
    if side < math.sqrt(11.0):
        return False

    # edge lengths within 1e-8 of 1 (unit hexagons)
    verts = _inner_vertices(positions, angles_deg)
    for i in range(len(positions)):
        v = verts[i]
        for k in range(6):
            d = v[(k + 1) % 6] - v[k]
            if abs(math.hypot(d[0], d[1]) - 1.0) > 1e-8:
                return False

    # outer vertices
    ov = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle_deg)
    # containment: every inner vertex inside outer hexagon
    for i in range(len(positions)):
        for p in verts[i]:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                e = p2 - p1
                cross = e[0] * (p[1] - p1[1]) - e[1] * (p[0] - p1[0])
                if cross < -1e-6:
                    return False

    # separation: for each pair, some edge normal separates them by > 1e-6
    normals = []
    for i in range(len(positions)):
        v = verts[i]
        for k in range(6):
            e = v[(k + 1) % 6] - v[k]
            nrm = np.array([-e[1], e[0]])
            nrm /= (np.linalg.norm(nrm) + 1e-30)
            normals.append(nrm)

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            vi = verts[i]
            vj = verts[j]
            ok = False
            for nrm in normals:
                pi = np.dot(vi, nrm)
                pj = np.dot(vj, nrm)
                if (pi.max() - pj.min() < -1e-6) or (pj.max() - pi.min() < -1e-6):
                    ok = True
                    break
            if not ok:
                return False
    return True


def _minimal_side(positions, angles_deg, outer_angle_deg):
    # minimal enclosing regular hexagon (fixed orientation) centered at origin
    verts = _inner_vertices(positions, angles_deg)
    a = math.radians(outer_angle_deg)
    normals = np.array([
        [math.cos(a + math.radians(60 * k)), math.sin(a + math.radians(60 * k))]
        for k in range(6)
    ])
    side = 0.0
    for nrm in normals:
        proj = verts.reshape(-1, 2) @ nrm
        side = max(side, proj.max(), -proj.min())
    return side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular
    hexagon, maximizing 1/outer_hex_side_length.

    Returns
        inner_hex_data: np.ndarray of shape (11,3), rows (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,) (x, y, angle_degrees).
        outer_hex_side_length: float.
    """
    # Rigid hexagonal cluster of 11 unit hexagons (all same orientation), centered at origin.
    # Coordinates are in units of the hexagon "radius" (center-to-vertex = 1).
    positions = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 1.7320508075688772],
        [-1.0, 1.7320508075688772],
        [-2.0, 0.0],
        [-1.0, -1.7320508075688772],
        [1.0, -1.7320508075688772],
        [3.0, 1.7320508075688772],
        [2.0, 3.4641016151377544],
        [0.0, 3.4641016151377544],
        [-1.0, 3.4641016151377544],
    ])
    angles_deg = np.zeros(11)

    # Objective: minimal enclosing regular hexagon side for a given rotation angle.
    def objective(theta_deg):
        return _minimal_side(positions, angles_deg, theta_deg)

    # Golden-section search over rotation angle in [0, 60) degrees.
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    lo = 0.0
    hi = 60.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc = objective(c)
    fd = objective(d)

    # Coarse pass: ~200 golden-section iterations.
    for _ in range(200):
        if time.time() > _DEADLINE:
            break
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - inv_phi * (hi - lo)
            fc = objective(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + inv_phi * (hi - lo)
            fd = objective(d)

    best_theta = (lo + hi) / 2.0
    best_side = objective(best_theta)

    # Refine around the best with a finer local search if time allows.
    if time.time() < _DEADLINE - 5.0:
        span = 2.0  # degrees
        t0 = best_theta - span
        t1 = best_theta + span
        c = t1 - inv_phi * (t1 - t0)
        d = t0 + inv_phi * (t1 - t0)
        fc = objective(c)
        fd = objective(d)
        for _ in range(120):
            if time.time() > _DEADLINE:
                break
            if fc < fd:
                t1 = d
                d = c
                fd = fc
                c = t1 - inv_phi * (t1 - t0)
                fc = objective(c)
            else:
                t0 = c
                c = d
                fc = fd
                d = t0 + inv_phi * (t1 - t0)
                fd = objective(d)
        cand_theta = (t0 + t1) / 2.0
        cand_side = objective(cand_theta)
        if cand_side < best_side:
            best_theta = cand_theta
            best_side = cand_side

    # Build the result.
    outer_center = np.array([0.0, 0.0])
    outer_angle_deg = best_theta
    side = float(best_side)

    inner_hex_data = np.column_stack([positions, angles_deg]).astype(float)
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle_deg], dtype=float)

    # Safety: if the computed side is slightly too small to be valid, nudge it up.
    if not _is_valid(positions, angles_deg, outer_center, outer_angle_deg, side):
        # increase side in small steps until valid (should rarely trigger)
        side = side * 1.0001
        for _ in range(200):
            if time.time() > _DEADLINE:
                break
            if _is_valid(positions, angles_deg, outer_center, outer_angle_deg, side):
                break
            side = side * 1.0001

    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END
