import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# ---- contact-graph definition for 11 unit hexagons (axial coords) ----
_AXIAL = [
    (0, 0),
    (1, 0), (2, 0), (3, 0),
    (0, 1), (1, 1), (2, 1), (3, 1),
    (1, 2), (2, 2), (3, 2),
]
# adjacency: axial neighbours (6 directions)
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
_adj = {i: [] for i in range(N)}
_axmap = {ax: i for i, ax in enumerate(_AXIAL)}
for i, (q, r) in enumerate(_AXIAL):
    for dq, dr in _DIRS:
        j = _axmap.get((q + dq, r + dr))
        if j is not None and j != i and j > i:
            _adj[i].append(j)

# distance between axial-neighbouring unit-hex centers
_ADJ_D = math.sqrt(3.0)

# precompute unit vectors between adjacent centers (in cluster frame)
_ADJ_UV = {}
for i in range(N):
    for j in _adj[i]:
        aq, ar = _AXIAL[i]
        bq, br = _AXIAL[j]
        ux = (bq - aq) + 0.5 * (br - ar)
        uy = (math.sqrt(3.0) / 2.0) * (br - ar)
        _ADJ_UV[(i, j)] = np.array([ux, uy])


def _hex_vertices(cx, cy, angle, size=1.0):
    verts = []
    for k in range(6):
        a = angle + k * math.pi / 3
        verts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    return np.array(verts)


def _cluster_centers(theta):
    """Centers of the 11 hexagons given global rotation theta (cluster centered at origin)."""
    d = _ADJ_D
    centers = np.empty((N, 2))
    for i in range(N):
        q, r = _AXIAL[i]
        x = d * (q + r / 2.0)
        y = d * (math.sqrt(3.0) / 2.0) * r
        centers[i] = [x, y]
    c = centers.mean(axis=0)
    centers = centers - c
    ct = math.cos(theta)
    st = math.sin(theta)
    R = np.array([[ct, -st], [st, ct]])
    return centers @ R.T


def _min_outer_side(points, outer_angle):
    k_angles = np.arange(6) * math.pi / 3 + outer_angle
    normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
    dots = points @ normals.T
    abs_dots = np.abs(dots)
    max_per_normal = np.max(abs_dots, axis=0)
    max_proj = np.max(max_per_normal)
    return (2.0 / math.sqrt(3)) * max_proj


def _separation_violation(centers, angles):
    """Sum of squared penetration depths between all pairs (0 if disjoint)."""
    n = len(centers)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            v1 = _hex_vertices(centers[i][0], centers[i][1], angles[i])
            v2 = _hex_vertices(centers[j][0], centers[j][1], angles[j])
            # SAT: find minimum overlap
            polys = [v1, v2]
            min_overlap = float('inf')
            separated = False
            for v in polys:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k + 1) % 6]
                    edge = p2 - p1
                    elen = np.linalg.norm(edge)
                    if elen < 1e-12:
                        continue
                    normal = np.array([-edge[1], edge[0]]) / elen
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    ov = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                    if ov <= 1e-9:
                        separated = True
                        break
                    min_overlap = min(min_overlap, ov)
                if separated:
                    break
            if not separated:
                total += min_overlap * min_overlap
    return total


def _objective(theta, outer_angle, repulse=100.0):
    """Penalty objective: outer side length + repulsion for overlaps."""
    centers = _cluster_centers(theta[0])
    angles = np.zeros(N)
    all_points = []
    for i in range(N):
        all_points.append(_hex_vertices(centers[i][0], centers[i][1], angles[i]))
    points_arr = np.vstack(all_points)
    S = _min_outer_side(points_arr, outer_angle)
    sep = _separation_violation(centers, angles)
    return S + repulse * sep


def _valid(centers, angles, outer_center, outer_side, outer_angle, eps=1e-9):
    verts_list = [_hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_overlap(verts_list[i], verts_list[j], eps):
                return False
    for i in range(N):
        for v in verts_list[i]:
            if not _in_outer(v, outer_center, outer_side, outer_angle):
                return False
    return True


def _poly_overlap(v1, v2, eps=1e-9):
    polys = [v1, v2]
    for v in polys:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                return False
    return True


def _in_outer(point, outer_center, outer_side, outer_angle):
    p = point - outer_center
    k_angles = np.arange(6) * math.pi / 3 + outer_angle
    normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
    dots = p @ normals.T
    apothem = outer_side * math.sqrt(3) / 2.0
    return np.all(np.abs(dots) <= apothem + 1e-9)


def _optimize_one(outer_angle, deadline, safety=8.0):
    """Run Nelder-Mead over global rotation theta for a fixed outer_angle."""
    # multiple initial rotations
    best_S = float('inf')
    best_theta = None
    best_centers = None
    best_angles = None
    t0 = time.time()
    for init_theta in [0.0, math.pi / 6, math.pi / 12, -math.pi / 6, math.pi / 3, math.pi / 4]:
        if time.time() > deadline - safety:
            break
        res = minimize(
            _objective,
            x0=np.array([init_theta]),
            args=(outer_angle, 100.0),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True},
        )
        theta = float(res.x[0])
        centers = _cluster_centers(theta)
        angles = np.zeros(N)
        all_points = []
        for i in range(N):
            all_points.append(_hex_vertices(centers[i][0], centers[i][1], angles[i]))
        points_arr = np.vstack(all_points)
        S = _min_outer_side(points_arr, outer_angle)
        # verify disjointness
        if _separation_violation(centers, angles) > 1e-6:
            continue
        # verify inside outer
        if not _valid(centers, angles, (0.0, 0.0), S, outer_angle):
            # try slightly larger S
            S2 = S * 1.001
            if _valid(centers, angles, (0.0, 0.0), S2, outer_angle):
                S = S2
            else:
                continue
        if S < best_S:
            best_S = S
            best_theta = theta
            best_centers = centers
            best_angles = angles
    if best_centers is None:
        return None
    return best_centers, best_angles, best_S, best_theta


def construct():
    start_time = time.time()
    deadline = 145.0
    safety = 8.0

    best_result = None
    best_S = float('inf')

    outer_angles = [0.0, math.pi / 6, math.pi / 12, -math.pi / 6, math.pi / 3, math.pi / 4]

    for oa in outer_angles:
        if time.time() > deadline - safety:
            break
        result = _optimize_one(oa, deadline, safety)
        if result is not None:
            c, a, S, theta = result
            if S < best_S:
                best_S = S
                best_result = (c, a, oa)

    if best_result is not None:
        c, a, oa = best_result
        inner = [[c[i][0], c[i][1], math.degrees(a[i])] for i in range(N)]
        return inner, (0.0, 0.0), best_S, math.degrees(oa)

    # Fallback: valid (loose) arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
