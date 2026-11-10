# EVOLVE-BLOCK-START
import time
import numpy as np

# ---------------- geometry helpers ----------------

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))


def _edge_normals(verts):
    # outward normals of each edge
    e = np.roll(verts, -1, axis=0) - verts
    # rotate edge vector by -90 deg -> (dy, -dx)
    n = np.column_stack((e[:, 1], -e[:, 0]))
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-300)
    # ensure outward: normal dot (vertex_mid - center) > 0
    mid = (verts + np.roll(verts, -1, axis=0)) / 2.0
    center = verts.mean(axis=0)
    d = (mid - center)
    sign = np.sign(d @ n)
    sign[sign == 0] = 1.0
    n = n * sign[:, None]
    return n


def _separation_ok(A, B, eps=1e-6):
    # A, B: (6,2) vertex arrays. Returns True if separated (gap>eps on some normal).
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    alln = np.vstack([nA, nB])
    # project
    pA = A @ alln.T
    pB = B @ alln.T
    minA = pA.min(axis=0)
    maxA = pA.max(axis=0)
    minB = pB.min(axis=0)
    maxB = pB.max(axis=0)
    # separated if minB - maxA > eps  OR  minA - maxB > eps for any normal
    return bool(((minB - maxA) > eps).any() or ((minA - maxB) > eps).any())


def _containment_ok(inner_verts, outer_verts, tol=-1e-6):
    # outer_verts (6,2) CCW. Check every inner vertex is left of each directed edge.
    e = np.roll(outer_verts, -1, axis=0) - outer_verts
    # cross(e, p - p1) >= tol for all
    # vectorized: for each edge i, cross(e_i, (P - p1_i))
    # cross(e, v) = e_x * v_y - e_y * v_x
    # shape: (6, M)
    e_x = e[:, 0, None]
    e_y = e[:, 1, None]
    p1 = outer_verts[:, :, None]
    P = inner_verts.T  # (2, M)
    vx = P[0][None, :] - p1[0]
    vy = P[1][None, :] - p1[1]
    cross = e_x * vy - e_y * vx
    return bool((cross >= tol).all())


def _compute_side(inner_verts, outer_verts):
    # side length of outer regular hexagon from its vertices
    edges = np.roll(outer_verts, -1, axis=0) - outer_verts
    return float(np.linalg.norm(edges, axis=1).mean())


def _validity_penalty(inner_verts, outer_verts, side):
    # returns 0 if fully valid, else positive penalty
    pen = 0.0
    n = inner_verts.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_ok(inner_verts[i], inner_verts[j]):
                pen += 1.0
    if not _containment_ok(inner_verts.reshape(-1, 2), outer_verts):
        pen += 10.0
    if side < np.sqrt(11.0) - 1e-9:
        pen += 5.0
    return pen


# ---------------- objective ----------------

def _objective(theta):
    # theta layout: [x0..x10, y0..y10, ang0..ang10, outer_ang]
    xs = theta[:11]
    ys = theta[11:22]
    angs = theta[22:33]
    outer_ang = theta[33]

    # initial outer side guess: large enough to contain; we will shrink via penalty
    # Use a side that scales with the bounding radius of inner hexes + margin
    centers = np.column_stack((xs, ys))
    radii = np.sqrt(11.0)  # circumradius of unit hex
    max_r = np.max(np.linalg.norm(centers, axis=1)) + radii
    # side of regular hex with circumradius R: R = side, so side ~ max_r
    s = max(max_r + 0.5, np.sqrt(11.0))

    # build outer
    outer_verts = _hex_vertices(0.0, 0.0, s, outer_ang)

    # build inner
    inner_verts = []
    for k in range(11):
        inner_verts.append(_hex_vertices(xs[k], ys[k], 1.0, angs[k]))
    inner_verts = np.array(inner_verts)  # (11,6,2)

    side = _compute_side(inner_verts, outer_verts)
    pen = _validity_penalty(inner_verts, outer_verts, side)
    # minimize side + heavy penalty for invalidity
    return side + 100.0 * pen


# ---------------- seed generator ----------------

def _make_seed(rng):
    # start from a reasonable hexagonal cluster
    base = np.array([
        [0, 0],
        [2.5, 0], [-2.5, 0],
        [1.25, 2.1650635094610964], [-1.25, 2.1650635094610964],
        [1.25, -2.1650635094610964], [-1.25, -2.1650635094610964],
        [3.75, 2.1650635094610964], [-3.75, 2.1650635094610964],
        [3.75, -2.1650635094610964], [-3.75, -2.1650635094610964],
    ])
    xs = base[:, 0] + rng.normal(0, 0.3, 11)
    ys = base[:, 1] + rng.normal(0, 0.3, 11)
    angs = np.zeros(11)  # keep aligned initially
    outer_ang = 0.0
    theta = np.concatenate([xs, ys, angs, [outer_ang]])
    return theta


def _build_result(theta):
    xs = theta[:11]
    ys = theta[11:22]
    angs = theta[22:33]
    outer_ang = theta[33]

    centers = np.column_stack((xs, ys))
    max_r = np.max(np.linalg.norm(centers, axis=1)) + np.sqrt(11.0)
    s = max(max_r + 0.5, np.sqrt(11.0))

    inner_hex_data = np.column_stack((xs, ys, angs))
    outer_hex_data = np.array([0.0, 0.0, outer_ang])
    return inner_hex_data, outer_hex_data, float(s)


def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 600.0
    # safety margin: stop optimizing 20s before deadline
    opt_deadline = t0 + 580.0

    rng = np.random.default_rng(12345)

    best_theta = None
    best_val = np.inf

    # Try a few restarts with Nelder-Mead
    from scipy.optimize import minimize

    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > opt_deadline:
            break
        theta0 = _make_seed(rng)
        try:
            res = minimize(
                _objective,
                theta0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-6,
                    'fatol': 1e-7,
                    'maxiter': 4000,
                    'maxfev': 8000,
                    'adaptive': True,
                },
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_theta = res.x
        except Exception:
            pass
        if time.time() > opt_deadline:
            break

    if best_theta is None:
        # fallback to seed
        best_theta = _make_seed(rng)

    # Final validity check; if invalid, fall back to a guaranteed-valid large container
    inner_hex_data, outer_hex_data, side = _build_result(best_theta)

    # verify
    xs = inner_hex_data[:, 0]
    ys = inner_hex_data[:, 1]
    angs = inner_hex_data[:, 2]
    outer_ang = outer_hex_data[2]
    s = side
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_ang)
    inner_verts = np.array([_hex_vertices(xs[k], ys[k], 1.0, angs[k]) for k in range(11)])
    valid = _validity_penalty(inner_verts, outer_verts, s) == 0.0

    if not valid:
        # Guaranteed valid fallback: use the seed grid with a big container
        fallback = np.array([
            [0, 0, 0],
            [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.1650635094610964, 0], [1.25, 2.1650635094610964, 0],
            [-1.25, -2.1650635094610964, 0], [1.25, -2.1650635094610964, 0],
            [-3.75, 2.1650635094610964, 0], [3.75, 2.1650635094610964, 0],
            [-3.75, -2.1650635094610964, 0], [3.75, -2.1650635094610964, 0],
        ])
        inner_hex_data, outer_hex_data, side = fallback, np.array([0.0, 0.0, 0.0]), 8.0

    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END
