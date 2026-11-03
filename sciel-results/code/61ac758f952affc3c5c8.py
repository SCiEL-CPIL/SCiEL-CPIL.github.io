import math
import time
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3.0)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-12

# The 6 face normals of a unit regular hexagon (pointy-top orientation, angle 0)
_NORMALS = np.array([
    [math.cos(math.pi / 6.0 + k * math.pi / 3.0),
     math.sin(math.pi / 6.0 + k * math.pi / 3.0)]
    for k in range(6)
])

def _hex_gap(a, b):
    """Minimum separation along the 6 face normals; >= SQRT3 means no overlap."""
    d = a - b
    projs = np.abs(d @ _NORMALS.T)
    return float(np.min(projs))

def _container_size(pts, beta):
    """
    Given centers (n,2) and container rotation beta, return
    (S, center, D) where D = max |projection| over the 6 container face normals
    and S = (D + APOTHEM)/APOTHEM.  Center is the midpoint of the 6 projection intervals.
    """
    normals = np.array([
        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
        for k in range(6)
    ])
    projs = pts @ normals.T  # (n, 6)
    maxp = np.max(projs, axis=0)
    minp = np.min(projs, axis=0)
    D = float(max(np.max(maxp), -np.min(minp)))
    # center: for each direction, midpoint; average over 6
    cx_vals = 0.5 * (maxp + minp)
    # Reconstruct a point whose projections match midpoints (least squares)
    # normals.T @ c = cx_vals  -> 6 eqs, 2 unknowns
    try:
        c = np.linalg.lstsq(normals.T, cx_vals, rcond=None)[0]
    except Exception:
        c = np.array([0.0, 0.0])
    S = (D + APOTHEM) / APOTHEM
    return S, (float(c[0]), float(c[1])), D

def _check_disjoint(pts, tol=1e-9):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if _hex_gap(pts[i], pts[j]) < SQRT3 - tol:
                return False
    return True

def _random_init(scale):
    """Random cluster of N centers within a disk of radius ~scale."""
    pts = np.zeros((N, 2))
    for i in range(N):
        r = scale * math.sqrt(np.random.rand())
        th = 2 * math.pi * np.random.rand()
        pts[i] = [r * math.cos(th), r * math.sin(th)]
    return pts

def _force_directed(pts, beta, steps=4000, dt=0.02, k_rep=50.0, k_att=0.02,
                    rep_margin=0.05, cool=0.9995):
    """
    Force-directed relaxation.  Returns (pts, beta, S).
    """
    pts = pts.copy()
    for it in range(steps):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        n = len(pts)
        forces = np.zeros_like(pts)

        # Repulsive forces between overlapping (or near-overlapping) pairs
        for i in range(n):
            for j in range(i + 1, n):
                gap = _hex_gap(pts[i], pts[j])
                if gap < SQRT3 + rep_margin:
                    # violation magnitude
                    viol = (SQRT3 + rep_margin) - gap
                    # direction: push apart along the normal that gave min projection
                    d = pts[i] - pts[j]
                    projs = np.abs(d @ _NORMALS.T)
                    k = int(np.argmin(projs))
                    nx, ny = _NORMALS[k]
                    # sign: if d . n < 0, flip
                    if d[0] * nx + d[1] * ny < 0:
                        nx, ny = -nx, -ny
                    f = k_rep * viol
                    forces[i] += f * np.array([nx, ny])
                    forces[j] -= f * np.array([nx, ny])

        # Attractive force pulling toward origin (shrinks container)
        for i in range(n):
            forces[i] -= k_att * pts[i]

        # Damped update
        pts += dt * forces
        dt *= cool

        # Keep beta fixed here; it's optimized separately
        S, center, D = _container_size(pts, beta)
        # Re-center pts so container center is at origin (removes drift)
        pts = pts - np.array(center)

    S, center, D = _container_size(pts, beta)
    return pts, beta, S

def _optimize_beta(pts, n_grid=72):
    """Grid + golden-section refinement of container rotation for fixed pts."""
    best_S = float('inf')
    best_beta = 0.0
    best_center = (0.0, 0.0)
    for i in range(n_grid):
        beta = i * (math.pi / 6.0) / (n_grid - 1)
        S, center, _ = _container_size(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center
    # Golden-section refine around best
    lo = max(0.0, best_beta - (math.pi / 6.0) / n_grid)
    hi = min(math.pi / 6.0, best_beta + (math.pi / 6.0) / n_grid)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a = lo + (1 - gr) * (hi - lo)
    b = lo + gr * (hi - lo)
    fa = _container_size(pts, a)[0]
    fb = _container_size(pts, b)[0]
    for _ in range(80):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if fa < fb:
            hi = b
            b = a
            fb = fa
            a = lo + (1 - gr) * (hi - lo)
            fa = _container_size(pts, a)[0]
        else:
            lo = a
            a = b
            fa = fb
            b = lo + gr * (hi - lo)
            fb = _container_size(pts, b)[0]
    beta = 0.5 * (lo + hi)
    S, center, _ = _container_size(pts, beta)
    if S < best_S:
        best_S = S
        best_beta = beta
        best_center = center
    return best_S, best_beta, best_center

def _joint_refine(pts, beta, center, steps=2000, dt=0.005, k_rep=30.0, k_att=0.01,
                  rep_margin=0.02):
    """
    Simultaneously refine positions AND container center via forces,
    with beta re-optimized periodically.  Returns (pts, beta, S).
    """
    pts = pts.copy()
    for it in range(steps):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        n = len(pts)
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                gap = _hex_gap(pts[i], pts[j])
                if gap < SQRT3 + rep_margin:
                    viol = (SQRT3 + rep_margin) - gap
                    d = pts[i] - pts[j]
                    projs = np.abs(d @ _NORMALS.T)
                    k = int(np.argmin(projs))
                    nx, ny = _NORMALS[k]
                    if d[0] * nx + d[1] * ny < 0:
                        nx, ny = -nx, -ny
                    f = k_rep * viol
                    forces[i] += f * np.array([nx, ny])
                    forces[j] -= f * np.array([nx, ny])
        # Pull toward container center
        for i in range(n):
            forces[i] -= k_att * (pts[i] - np.array(center))
        pts += dt * forces
        if it % 200 == 0:
            # Re-optimize beta and re-center
            _, beta, _ = _optimize_beta(pts, n_grid=48)
            S, center, _ = _container_size(pts, beta)
    S, center, _ = _container_size(pts, beta)
    return pts, beta, S

def _lattice_seed():
    """A decent hexagonal cluster seed (center + 6 + 4 more)."""
    pts = []
    pts.append((0.0, 0.0))
    for k in range(6):
        ang = math.pi / 6 + k * math.pi / 3
        pts.append((SQRT3 * math.cos(ang), SQRT3 * math.sin(ang)))
    # 4 more at radius ~2*SQRT3
    for k in [0, 2, 4, 1]:
        ang = math.pi / 6 + k * math.pi / 3
        pts.append((2 * SQRT3 * math.cos(ang), 2 * SQRT3 * math.sin(ang)))
    return np.array(pts[:N])

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = (0.0, 0.0)

    def _update(pts, beta):
        nonlocal best_S, best_pts, best_beta, best_center
        S, center, _ = _container_size(pts, beta)
        if S < best_S and _check_disjoint(pts):
            best_S = S
            best_pts = pts.copy()
            best_beta = beta
            best_center = center

    # --- Phase 1: lattice seed + force-directed ---
    seed = _lattice_seed()
    for beta_init in [0.0, math.pi / 12.0, math.pi / 18.0]:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts_f, beta_f, S_f = _force_directed(seed, beta_init, steps=3000)
        S_b, beta_b, center_b = _optimize_beta(pts_f)
        _update(pts_f, beta_b)

    # --- Phase 2: many random restarts with force-directed ---
    rng = np.random.default_rng(42)
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        scale = 3.0 + 2.0 * rng.random()
        pts0 = _random_init(scale)
        # jitter
        pts0 += 0.3 * rng.standard_normal((N, 2))
        beta0 = rng.random() * (math.pi / 6.0)
        pts_f, beta_f, S_f = _force_directed(pts0, beta0, steps=2500)
        S_b, beta_b, center_b = _optimize_beta(pts_f)
        _update(pts_f, beta_b)

    # --- Phase 3: joint refinement on best (if any) ---
    if best_pts is not None:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            pass
        else:
            pts_j, beta_j, S_j = _joint_refine(best_pts, best_beta, best_center,
                                               steps=3000, dt=0.003)
            S_b, beta_b, center_b = _optimize_beta(pts_j)
            _update(pts_j, beta_b)

        # A few more joint refinements from perturbed best
        for _ in range(5):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pert = best_pts + 0.1 * np.random.default_rng().standard_normal((N, 2))
            pts_j, beta_j, S_j = _joint_refine(pert, best_beta, best_center,
                                               steps=1500, dt=0.003)
            S_b, beta_b, center_b = _optimize_beta(pts_j)
            _update(pts_j, beta_b)

    # --- Phase 4: final high-precision polish of beta and center ---
    if best_pts is not None:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 3:
            S_b, beta_b, center_b = _optimize_beta(best_pts, n_grid=200)
            if S_b < best_S:
                best_S = S_b
                best_beta = beta_b
                best_center = center_b

    # Fallback
    if best_pts is None:
        best_pts = _lattice_seed()
        best_beta = 0.0
        best_S, best_center, _ = _container_size(best_pts, best_beta)

    # Final validation: ensure disjoint; if not, scale apart slightly
    if not _check_disjoint(best_pts, tol=1e-8):
        # gentle expansion from centroid until disjoint (rare)
        cen = np.mean(best_pts, axis=0)
        for _ in range(50):
            best_pts = cen + 1.001 * (best_pts - cen)
            if _check_disjoint(best_pts, tol=1e-9):
                break
        best_S, best_center, _ = _container_size(best_pts, best_beta)

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    # Recompute exact container from final pts and beta
    S_final, center_final, D_final = _container_size(best_pts, best_beta)
    cx, cy = center_final

    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
