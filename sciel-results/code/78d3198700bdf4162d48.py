import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
SQRT3 = math.sqrt(3)

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]


def _hex_vertices(pos, ang_rad):
    """Return 6x2 vertices of a unit hexagon at pos with rotation ang_rad."""
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    vx = pos[0] + _UX * c - _UY * s
    vy = pos[1] + _UX * s + _UY * c
    return np.column_stack((vx, vy))


def _edge_normals(ang_rad):
    """Return 6x2 outward edge normals for a hexagon rotated by ang_rad."""
    # Original edge normals (computed from unit hex edges)
    en = np.zeros((6, 2))
    for i in range(6):
        ex = _UX[(i + 1) % 6] - _UX[i]
        ey = _UY[(i + 1) % 6] - _UY[i]
        nx, ny = -ey, ex
        norm = math.sqrt(nx * nx + ny * ny)
        en[i] = [nx / norm, ny / norm]
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return en @ np.array([[c, -s], [s, c]]).T


def _pairwise_overlap_penalty(v1, v2, beta=50.0):
    """
    Compute a smooth overlap penalty between two convex hexagons.
    Uses the separating axis theorem in a soft form:
    For each of the 12 candidate normals (6 from each hex), compute the
    minimum separation. If all separations are <= 0, they overlap.
    We use a soft-min of the separations (negative means overlap).
    """
    # v1, v2 are 6x2 arrays of vertices
    # Compute edges and normals for each
    def get_normals(verts):
        normals = np.zeros((6, 2))
        for i in range(6):
            ex = verts[(i + 1) % 6, 0] - verts[i, 0]
            ey = verts[(i + 1) % 6, 1] - verts[i, 1]
            nx, ny = -ey, ex
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-12:
                normals[i] = [1.0, 0.0]
            else:
                normals[i] = [nx / norm, ny / norm]
        return normals

    n1 = get_normals(v1)
    n2 = get_normals(v2)

    min_seps = []
    for nx, ny in np.vstack([n1, n2]):
        proj1 = v1[:, 0] * nx + v1[:, 1] * ny
        proj2 = v2[:, 0] * nx + v2[:, 1] * ny
        sep = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
        # sep > 0 means separated, sep < 0 means overlapping
        min_seps.append(sep)

    min_seps = np.array(min_seps)
    # Soft min: if all min_seps > 0, penalty ~ 0. If any is negative, penalty grows.
    # Use: penalty = -softmin(min_seps) where softmin uses log-sum-exp
    # Actually we want: if min_seps has negative values, penalize.
    # Use: penalty = max(0, -min(min_seps)) but smoothed.
    # Better: use the minimum separation directly, penalize if negative.
    min_sep = np.min(min_seps)
    if min_sep < 0:
        return -min_sep  # linear penalty for overlap depth
    else:
        return 0.0


def _soft_pairwise_overlap(v1, v2, beta=30.0):
    """
    Differentiable soft overlap penalty using log-sum-exp of negative separations.
    Returns a smooth approximation of the maximum penetration depth.
    """
    def get_normals(verts):
        normals = np.zeros((6, 2))
        for i in range(6):
            ex = verts[(i + 1) % 6, 0] - verts[i, 0]
            ey = verts[(i + 1) % 6, 1] - verts[i, 1]
            nx, ny = -ey, ex
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-12:
                normals[i] = [1.0, 0.0]
            else:
                normals[i] = [nx / norm, ny / norm]
        return normals

    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])

    seps = np.zeros(len(all_normals))
    for k, (nx, ny) in enumerate(all_normals):
        proj1 = v1[:, 0] * nx + v1[:, 1] * ny
        proj2 = v2[:, 0] * nx + v2[:, 1] * ny
        seps[k] = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())

    # We want to penalize when min(seps) < 0
    # Use soft min: -logsumexp(-seps / beta) / beta
    # This gives a smooth lower bound on min(seps)
    # If min(seps) is very negative, this approximates min(seps)
    # If min(seps) is positive, this is close to 0 (slightly negative)
    # We want penalty = max(0, -min_seps) smoothed
    # Let's use: penalty = -softmin(seps) where softmin = -logsumexp(-seps/beta)/beta
    # Then penalty = logsumexp(-seps/beta)/beta
    # But this is always positive. We want 0 when separated.
    # Alternative: use the raw min_sep and apply a smooth rectifier
    min_sep = np.min(seps)
    # Smooth rectifier: f(x) = x + log(1 + exp(-x/beta)) / beta, for x = -min_sep
    x = -min_sep  # positive when overlapping
    if x > 50.0 / beta:
        return x
    elif x < -50.0 / beta:
        return 0.0
    else:
        return x + math.log(1.0 + math.exp(-x * beta)) / beta


def _energy(params, outer_side, beta=30.0):
    """
    params: array of shape (N, 3) = [x, y, angle_rad] for each hexagon
    outer_side: side length of the outer regular hexagon
    Returns: total energy (collision + boundary)
    """
    # Compute vertices for all hexagons
    verts_list = []
    for i in range(N):
        pos = params[i, :2]
        ang = params[i, 2]
        verts_list.append(_hex_vertices(pos, ang))

    # Pairwise collision penalty
    collision = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Quick distance check
            d2 = np.sum((params[i, :2] - params[j, :2]) ** 2)
            if d2 > 4.5:
                continue
            collision += _soft_pairwise_overlap(verts_list[i], verts_list[j], beta)

    # Boundary containment: all vertices must be inside the outer hexagon
    # Outer hexagon is centered at origin with side length outer_side
    # A point (x,y) is inside a regular hexagon of side s centered at origin if:
    # For each of the 6 edge normals, the projection is within the inradius
    # Inradius of regular hexagon with side s is s * sqrt(3) / 2
    inradius = outer_side * SQRT3 / 2.0

    boundary = 0.0
    for i in range(N):
        verts = verts_list[i]
        for k in range(6):
            # Edge normal direction for the outer hexagon (flat-top orientation)
            # Outer hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
            # Edge normals point outward at angles 30, 90, 150, 210, 270, 330
            theta = np.pi / 6 + k * np.pi / 3
            nx, ny = math.cos(theta), math.sin(theta)
            proj = verts[:, 0] * nx + verts[:, 1] * ny
            max_proj = np.max(proj)
            if max_proj > inradius:
                boundary += (max_proj - inradius) ** 2

    return collision + 10.0 * boundary


def _energy_grad(params, outer_side, beta=30.0):
    """Numerical gradient of energy."""
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = _energy(params, outer_side, beta)
    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            params[i, j] += eps
            fp = _energy(params, outer_side, beta)
            params[i, j] -= 2 * eps
            fm = _energy(params, outer_side, beta)
            params[i, j] += eps
            grad[i, j] = (fp - fm) / (2 * eps)
    return grad


def _is_valid_config(params):
    """Check if configuration is valid (no overlaps, all inside a hexagon of some side)."""
    # First check pairwise disjointness using SAT
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(params[i, :2], params[i, 2]))

    for i in range(N):
        for j in range(i + 1, N):
            if not _sat_disjoint(verts_list[i], verts_list[j]):
                return False
    return True


def _sat_disjoint(v1, v2):
    """SAT check: return True if v1 and v2 are disjoint."""
    def get_normals(verts):
        normals = np.zeros((6, 2))
        for i in range(6):
            ex = verts[(i + 1) % 6, 0] - verts[i, 0]
            ey = verts[(i + 1) % 6, 1] - verts[i, 1]
            nx, ny = -ey, ex
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-12:
                normals[i] = [1.0, 0.0]
            else:
                normals[i] = [nx / norm, ny / norm]
        return normals

    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    for nx, ny in all_normals:
        proj1 = v1[:, 0] * nx + v1[:, 1] * ny
        proj2 = v2[:, 0] * nx + v2[:, 1] * ny
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return True
    return False


def _compute_min_outer_side(params):
    """Given valid params, compute the minimum enclosing regular hexagon side length."""
    # Collect all vertices
    all_verts = np.vstack([_hex_vertices(params[i, :2], params[i, 2]) for i in range(N)])
    center = np.mean(all_verts, axis=0)
    offsets = all_verts - center

    def eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        return float(np.max(side_vals))

    # Coarse scan
    phi_vals = np.arange(0, 60, 0.5)
    best_side = float('inf')
    best_phi = 0.0
    for p in phi_vals:
        s = eval_phi(float(p))
        if s < best_side:
            best_side = s
            best_phi = float(p)

    # Fine tune
    lo = max(0.0, best_phi - 0.5)
    hi = min(60.0, best_phi + 0.5)
    gr = (math.sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = eval_phi(x1)
    f2 = eval_phi(x2)
    for _ in range(40):
        if f1 < f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - gr * (hi - lo)
            f1 = eval_phi(x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + gr * (hi - lo)
            f2 = eval_phi(x2)
    best_phi = (lo + hi) / 2.0
    best_side = min(best_side, eval_phi(best_phi))

    return best_side, best_phi, float(center[0]), float(center[1])


def _make_honeycomb_params():
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    params = np.zeros((N, 3))
    for idx, (i, j) in enumerate(lattice_coords):
        params[idx, 0] = i * a1[0] + j * a2[0]
        params[idx, 1] = i * a1[1] + j * a2[1]
        params[idx, 2] = 0.0
    return params


def _make_ring_params():
    params = np.zeros((N, 3))
    for i in range(6):
        theta = i * np.pi / 3
        params[i, 0] = 2.5 * SQRT3 * math.cos(theta)
        params[i, 1] = 2.5 * SQRT3 * math.sin(theta)
    for i in range(6):
        theta = (i + 0.5) * np.pi / 3
        params[6 + i, 0] = 1.2 * SQRT3 * math.cos(theta)
        params[6 + i, 1] = 1.2 * SQRT3 * math.sin(theta)
    return params


def _make_two_rows_params():
    params = np.zeros((N, 3))
    for i in range(6):
        params[i, 0] = i * SQRT3
        params[i, 1] = 0.0
    for i in range(6):
        params[6 + i, 0] = i * SQRT3 + SQRT3 / 2
        params[6 + i, 1] = 1.5
    return params


def _make_random_params(rng, scale=3.0):
    params = np.zeros((N, 3))
    params[:, 0] = rng.uniform(-scale, scale, N)
    params[:, 1] = rng.uniform(-scale, scale, N)
    params[:, 2] = rng.uniform(-np.pi / 6, np.pi / 6, N)
    return params


def _gradient_descent(params0, outer_side, time_limit, rng):
    """Run L-BFGS-B to minimize energy for a fixed outer_side."""
    t_start = time.time()
    beta = 30.0

    result = minimize(
        _energy,
        params0,
        args=(outer_side, beta),
        method='L-BFGS-B',
        jac=_energy_grad,
        options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12},
    )

    return result.x, result.fun


def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)

    # Start with a reasonable outer side estimate
    # Honeycomb has a known side length
    honey = _make_honeycomb_params()
    h_side, h_phi, h_cx, h_cy = _compute_min_outer_side(honey)
    print(f"Honeycomb initial side: {h_side:.4f}")

    best_side = float('inf')
    best_params = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    inits = [
        _make_honeycomb_params(),
        _make_ring_params(),
        _make_two_rows_params(),
        _make_random_params(rng, 3.0),
        _make_random_params(rng, 2.5),
        _make_random_params(rng, 3.5),
    ]

    for init_idx, params0 in enumerate(inits):
        elapsed = time.time() - t0
        if elapsed > DEADLINE - 20.0:
            break

        # Try a few target outer side lengths, starting from the honeycomb estimate
        target_sides = [h_side, h_side * 0.95, h_side * 0.90, h_side * 0.85]

        for target_side in target_sides:
            elapsed = time.time() - t0
            if elapsed > DEADLINE - 10.0:
                break

            # Run gradient descent
            try:
                params_opt, energy = _gradient_descent(params0, target_side, 15.0, rng)
            except Exception:
                continue

            # Check if the optimized config is actually valid (disjoint)
            if not _is_valid_config(params_opt):
                continue

            # Check if all hexagons fit inside a hexagon of side target_side
            # by computing the actual min enclosing side
            actual_side, phi, cx, cy = _compute_min_outer_side(params_opt)

            if actual_side < best_side:
                best_side = actual_side
                best_params = params_opt.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

            elapsed = time.time() - t0
            if elapsed > DEADLINE - 10.0:
                break

    # If no valid config found, fall back to honeycomb
    if best_params is None:
        best_params = honey.copy()
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_params)

    # Final verification
    if not _is_valid_config(best_params):
        best_params = honey.copy()
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_params)

    inner = []
    for i in range(N):
        inner.append([float(best_params[i, 0]), float(best_params[i, 1]), float(math.degrees(best_params[i, 2]))])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
