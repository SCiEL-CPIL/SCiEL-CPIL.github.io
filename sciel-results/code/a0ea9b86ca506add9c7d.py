import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# 6 outward normals of a unit regular hexagon (apothem = sqrt(3)/2)
_NORMALS = np.array([
    [math.cos(math.pi / 6.0 + k * math.pi / 3.0),
     math.sin(math.pi / 6.0 + k * math.pi / 3.0)]
    for k in range(6)
])
# Minkowski-sum apothem for two unit hexagons with same orientation = sqrt(3)
_MINK_APO = math.sqrt(3)


def _hex_normals(beta):
    """Outward normals of a regular hexagon rotated by beta (radians)."""
    return np.array([
        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
        for k in range(6)
    ])


def _min_enclosing_hexagon(pts, beta):
    """
    Return (S, center, D) for the smallest regular hexagon (apothem D, side
    S=(D+APOTHEM)/APOTHEM) with rotation beta that contains all points in pts.
    Uses a 2D LP: minimize D s.t. -(D) <= (p_i - c) . n_k <= D for all i,k.
    """
    n = len(pts)
    normals = _hex_normals(beta)
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in pts:
            # (px-cx)*nx + (py-cy)*ny <= D  ->  -nx*cx - ny*cy - D <= -px*nx - py*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            # (px-cx)*nx + (py-cy)*ny >= -D -> nx*cx + ny*cy - D <= px*nx + py*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    from scipy.optimize import linprog
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.status == 0:
            D = float(res.x[2])
            cx = float(res.x[0])
            cy = float(res.x[1])
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy), D
    except Exception:
        pass
    return float('inf'), None, None


def _check_no_overlap(pts):
    """All inner hexagons share orientation; non-overlap iff for every pair
    and every of the 6 normals, |d . n| >= sqrt(3)."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            projs = np.abs(d @ _NORMALS.T)
            if np.min(projs) < _MINK_APO - 1e-9:
                return False
    return True


def _force_directed_pack(init_pts, max_iter=4000, dt0=0.02, dt_min=1e-6,
                         k_rep=5.0, k_att=0.5, cool=0.9995, seed=0):
    """
    Continuous packing: hexagons are particles.  Repulsive force when Minkowski
    distance < _MINK_APO (they overlap), attractive force when > _MINK_APO
    (they are separated).  This drives the system to a taut, contact-rich
    configuration.  Returns (pts, S, beta, center) with the best S seen.
    """
    rng = np.random.default_rng(seed)
    pts = init_pts.astype(float).copy()
    n = len(pts)
    best_S = float('inf')
    best_pts = pts.copy()
    best_beta = 0.0
    best_center = None

    # Beta symmetry period is pi/6; we'll search beta in [0, pi/6).
    # We'll interleave beta sweeps with force iterations.

    dt = dt0
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        # --- Force computation ---
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    # Jitter to break symmetry
                    d = rng.normal(size=2) * 1e-3
                    dist = np.linalg.norm(d)
                # Minkowski "distance" in the hexagonal metric:
                # min over 6 normals of |d . n|
                projs = np.abs(d @ _NORMALS.T)
                min_proj = np.min(projs)
                # Identify which normal is the active one
                k_active = int(np.argmin(projs))
                n_active = _NORMALS[k_active]
                # Signed distance along active normal
                sgn = np.sign(d @ n_active)
                if sgn == 0:
                    sgn = 1.0
                # Gap: positive means separated, negative means overlapping
                gap = min_proj - _MINK_APO
                if gap < 0:
                    # Overlap: repel strongly
                    f = k_rep * (-gap) + 0.1
                    f_dir = sgn * n_active
                else:
                    # Separated: attract gently (only if gap is small)
                    if gap < 2.0:
                        f = k_att * gap
                        f_dir = -sgn * n_active
                    else:
                        f = 0.0
                        f_dir = np.zeros(2)
                forces[i] += f * f_dir
                forces[j] -= f * f_dir

        # --- Small random perturbation to escape local minima ---
        forces += rng.normal(size=pts.shape) * 0.005

        # --- Integrate ---
        pts += dt * forces
        dt = max(dt * cool, dt_min)

        # --- Evaluate S for several betas (cheap grid) ---
        if it % 5 == 0:
            for bi in range(13):
                beta = bi * (math.pi / 6.0) / 12.0
                S, center, D = _min_enclosing_hexagon(pts, beta)
                if S < best_S and _check_no_overlap(pts):
                    best_S = S
                    best_pts = pts.copy()
                    best_beta = beta
                    best_center = center

    # Final refinement: fine beta sweep on best_pts
    if best_center is not None:
        for bi in range(121):
            beta = bi * (math.pi / 6.0) / 120.0
            S, center, D = _min_enclosing_hexagon(best_pts, beta)
            if S < best_S and _check_no_overlap(best_pts):
                best_S = S
                best_beta = beta
                best_center = center
        # minimize_scalar on beta
        from scipy.optimize import minimize_scalar
        try:
            res = minimize_scalar(
                lambda b: _min_enclosing_hexagon(best_pts, b)[0],
                bounds=(0, math.pi / 6.0), method='bounded',
                options={'maxiter': 200, 'xatol': 1e-14})
            if res.fun < best_S:
                S2, c2, _ = _min_enclosing_hexagon(best_pts, res.x)
                if _check_no_overlap(best_pts):
                    best_S = S2
                    best_beta = float(res.x)
                    best_center = c2
        except Exception:
            pass

    return best_pts, best_S, best_beta, best_center


def _local_refine(pts, beta, center):
    """
    Nelder-Mead local refinement of positions (keeping beta and center
    re-optimized via LP each call).  Objective = S + overlap penalty.
    """
    from scipy.optimize import minimize

    def obj(params):
        p = params.reshape(N, 2)
        if not _check_no_overlap(p):
            # Penalty
            pen = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = p[i] - p[j]
                    projs = np.abs(d @ _NORMALS.T)
                    min_proj = np.min(projs)
                    if min_proj < _MINK_APO:
                        pen += (_MINK_APO - min_proj) * 100.0
            S, _, _ = _min_enclosing_hexagon(p, beta)
            return S + pen
        S, _, _ = _min_enclosing_hexagon(p, beta)
        return S

    x0 = pts.ravel()
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
        p = res.x.reshape(N, 2)
        if _check_no_overlap(p):
            S, c, _ = _min_enclosing_hexagon(p, beta)
            if S < float('inf'):
                return p, S, beta, c
    except Exception:
        pass
    return pts, float('inf'), beta, None


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Initial configurations: diverse random + a few structured starts ---
    rng = np.random.default_rng(42)

    def _structured_start(kind):
        """Generate a set of N=11 points in a compact arrangement."""
        pts = []
        if kind == 0:
            # Hexagonal spiral
            axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
                     (2,0),(-2,0),(0,2),(1,-2)]
        elif kind == 1:
            # 3x4 grid-ish
            axial = [(0,0),(1,0),(2,0),(3,0),
                     (0,1),(1,1),(2,1),
                     (0,2),(1,2),(2,2),
                     (0,3)]
        elif kind == 2:
            # 2-wide column
            axial = [(0,0),(1,0),
                     (0,1),(1,1),(2,1),
                     (0,2),(1,2),(2,2),
                     (0,3),(1,3),
                     (0,4)]
        else:
            axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
                     (1,1),(-1,2),(-2,1),(2,-1)]
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    starts = []
    for k in range(4):
        starts.append(_structured_start(k))
    # Random starts
    for _ in range(8):
        pts = rng.normal(size=(N, 2)) * 2.0
        starts.append(pts)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for s_idx, init_pts in enumerate(starts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        # Force-directed packing
        pts_fd, S_fd, beta_fd, center_fd = _force_directed_pack(
            init_pts, max_iter=2000, seed=s_idx * 100 + 7)

        if S_fd < best_global_S and center_fd is not None:
            best_global_S = S_fd
            best_global_pts = pts_fd
            best_global_beta = beta_fd
            best_global_center = center_fd

        # Local refinement on the force-directed result
        if center_fd is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            pts_lr, S_lr, beta_lr, center_lr = _local_refine(pts_fd, beta_fd, center_fd)
            if S_lr < best_global_S and center_lr is not None:
                best_global_S = S_lr
                best_global_pts = pts_lr
                best_global_beta = beta_lr
                best_global_center = center_lr

        # Early exit if we're doing very well
        if best_global_S < 3.4:
            break

    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        pts = _structured_start(0)
        best_global_pts = pts
        best_global_beta = 0.0
        best_global_S, best_global_center, _ = _min_enclosing_hexagon(pts, 0.0)
        if best_global_center is None:
            best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

    # Final polish: one more local refinement pass
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_global_center is not None:
        pts_f, S_f, beta_f, center_f = _local_refine(
            best_global_pts, best_global_beta, best_global_center)
        if S_f < best_global_S and center_f is not None:
            best_global_S = S_f
            best_global_pts = pts_f
            best_global_beta = beta_f
            best_global_center = center_f

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Tiny scale-up for safety against numerical overlap
    SCALE = 1.0 + 1e-10
    pts_scaled = np.array([
        [cx + (x - cx) * SCALE, cy + (y - cy) * SCALE]
        for x, y in best_global_pts
    ])

    # Recompute exact S for the scaled points
    max_proj = 0.0
    normals = _hex_normals(best_global_beta)
    for nx, ny in normals:
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = [[float(x), float(y), float(angle_deg)] for x, y in pts_scaled]
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
