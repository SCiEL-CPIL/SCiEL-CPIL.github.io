"""Physics-based repulsion + annealing for packing 12 unit hexagons."""
import math
import time
import numpy as np

N = 12
T_START = time.time()
DEADLINE = T_START + 130.0  # safety margin before 150s kill

# 6 face normals of a unit regular hexagon, flat-top orientation (angle=0)
NORMALS = np.array([[math.cos(k * math.pi / 3.0) for k in range(6)]], dtype=float)  # (1,6)
NORMALS = NORMALS[0]  # (6,)
VERTS_UNIT = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 7)[:-1]])  # (6,2)


def _hex_verts(center, angle_deg):
    """Vertices (6,2) of a unit hexagon at center, rotation angle (deg)."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    return center + (VERTS_UNIT @ R.T)


def _face_dist(verts, normals):
    """(6,) max over vertices of dot(vert - origin, normal) for each face normal."""
    return verts @ normals  # (6,)


def _min_side(center, angle_deg):
    """Minimal outer side length (origin-centered, flat-top) containing this hexagon."""
    verts = _hex_verts(center, angle_deg)
    return np.max(verts @ NORMALS)  # scalar


def _min_side_all(centers, angles):
    """(N,) minimal outer side length for each hexagon (origin-centered outer)."""
    verts = centers + (VERTS_UNIT @ np.stack(
        [np.array([math.cos(math.radians(a)), math.sin(math.radians(a))]) for a in angles]).T).T
    return np.max(verts @ NORMALS, axis=1)


def _overlap_penalty(centers, angles, scale=1.0):
    """Sum of squared overlap depths between all pairs (smooth-ish surrogate)."""
    pen = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = centers[i] - centers[j]
            dist = math.hypot(d[0], d[1])
            # overlap depth if centers closer than 2 (max possible inscribed-circle sum)
            if dist < 2.0:
                pen += (2.0 - dist) ** 2 * scale
    return pen


def _run(centers, angles, T0, steps, dt, T_end, outer_center, outer_side, outer_angle):
    """Anneal positions+angles to minimize overlaps + containment. Returns (centers, angles, final_energy)."""
    # outer hexagon face normals (origin-centered then shifted)
    th = math.radians(outer_angle)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    outer_normals = NORMALS @ R.T  # (6,)
    # containment: for each hexagon, max over its vertices of (dot(v - outer_center, n) - outer_side)
    # We want all <= 0. Penalty = sum of positive violations squared.

    def energy(centers, angles):
        e = 0.0
        # overlap
        for i in range(N):
            for j in range(i + 1, N):
                d = centers[i] - centers[j]
                dist = math.hypot(d[0], d[1])
                if dist < 2.0:
                    e += (2.0 - dist) ** 2
        # containment
        for i in range(N):
            th_i = math.radians(angles[i])
            ci, si = math.cos(th_i), math.sin(th_i)
            Ri = np.array([[ci, -si], [si, ci]])
            verts = centers[i] + (VERTS_UNIT @ Ri.T)
            # distance from outer_center to each face: dot(vert - oc, n) ; must be <= side
            viol = np.max((verts - outer_center) @ outer_normals - outer_side)
            if viol > 0:
                e += 10.0 * viol ** 2
        return e

    best_e = energy(centers, angles)
    best_c = centers.copy()
    best_a = angles.copy()
    T = T0
    for step in range(steps):
        if time.time() > DEADLINE:
            break
        # random perturbation
        dpos = np.random.randn(N, 2) * dt
        dang = np.random.randn(N) * dt * 0.5
        new_c = centers + dpos
        new_a = np.clip(angles + dang, -180, 180)
        e_new = energy(new_c, new_a)
        if e_new < best_e or np.random.rand() < math.exp(-(e_new - best_e) / max(T, 1e-6)):
            centers, angles = new_c, new_a
            if e_new < best_e:
                best_e = e_new
                best_c = centers.copy()
                best_a = angles.copy()
        T = max(T_end, T * (T_end / T0) ** (1.0 / max(steps, 1)))
    return best_c, best_a, best_e


def construct():
    T0 = 5.0
    T_end = 0.01
    dt = 0.15
    steps_per_run = 400
    # Coarse-to-fine outer side: start loose, tighten
    best = None
    best_side = float('inf')
    restarts = 30
    for r in range(restarts):
        if time.time() > DEADLINE:
            break
        # random initial layout: place hexagons in a loose cluster
        centers = np.random.randn(N, 2) * 3.0
        angles = np.random.uniform(-30, 30, N)
        outer_center = np.zeros(2)
        outer_angle = 0.0
        # estimate initial outer side
        est_side = max(2.0, np.max(_min_side_all(centers, angles)) + 0.5)
        # try a sequence of outer sides decreasing
        for trial in range(6):
            if time.time() > DEADLINE:
                break
            side = est_side * (0.85 ** trial)
            if side < 2.0:
                side = 2.0
            c2, a2, e = _run(centers, angles, T0, steps_per_run, dt, T_end, outer_center, side, outer_angle)
            # check validity: compute actual minimal side needed
            needed = np.max(_min_side_all(c2, a2))
            # also verify no overlap (dist >= 2 - tol)
            ok = True
            for i in range(N):
                for j in range(i + 1, N):
                    if math.hypot(c2[i][0] - c2[j][0], c2[i][1] - c2[j][1]) < 2.0 - 1e-3:
                        ok = False
                        break
                if not ok:
                    break
            if ok and needed <= side + 1e-3:
                # valid packing with outer side = needed (tighten)
                if needed < best_side:
                    best_side = needed
                    best = (c2.copy(), a2.copy(), outer_center.copy(), outer_angle)
                # tighten further using found layout
                centers, angles = c2, a2
                est_side = needed
            else:
                # infeasible at this side; keep last feasible centers for next tighter trial
                # but don't update est_side downward if infeasible
                break
    if best is None:
        # fallback: return something (shouldn't happen)
        centers = np.zeros((N, 2))
        angles = np.zeros(N)
        return [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)], (0.0, 0.0), 40.0, 0.0
    c2, a2, oc, oa = best
    inner = [[float(c2[i][0]), float(c2[i][1]), float(a2[i])] for i in range(N)]
    return inner, (float(oc[0]), float(oc[1])), float(best_side), float(oa)
