import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 140.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]
_SQRT3 = math.sqrt(3)


def _hexagons_disjoint_fast(h1, h2):
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5:
        return True
    a1 = h1[2]
    a2 = h2[2]
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    v1x = h1[0] + _UX * c1 - _UY * s1
    v1y = h1[1] + _UX * s1 + _UY * c1
    v2x = h2[0] + _UX * c2 - _UY * s2
    v2y = h2[1] + _UX * s2 + _UY * c2
    en1x = _ENX * c1 - _ENY * s1
    en1y = _ENX * s1 + _ENY * c1
    en2x = _ENX * c2 - _ENY * s2
    en2y = _ENX * s2 + _ENY * c2
    for i in range(6):
        nx, ny = en1x[i], en1y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    for i in range(6):
        nx, ny = en2x[i], en2y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    return False


def _is_valid(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hexagons_disjoint_fast(h1, h2):
                return False
    return True


def _compute_min_outer_side(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    def eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / _SQRT3) * max_projs
        return np.max(side_vals)

    # Vectorized coarse scan
    phi_vals = np.radians(np.arange(0, 60, 1))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)  # (60, 6)
    side_vals = (2.0 / _SQRT3) * max_projs
    best_side = np.max(side_vals).min()
    best_idx = np.argmin(np.max(side_vals, axis=1))
    best_phi = float(best_idx)

    # Fine tune around best
    try:
        lo = max(0.0, best_phi - 1.0)
        hi = min(60.0, best_phi + 1.0)
        res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': 100})
        if res.success and res.fun < best_side:
            best_side = float(res.fun)
            best_phi = float(res.x)
    except Exception:
        pass

    return float(best_side), float(best_phi), float(center[0]), float(center[1])


def _smooth_objective(params, beta=8.0):
    """
    params: (3N,) = [x0,y0,a0, x1,y1,a1, ...]
    Returns: outer_side + penalty * overlap_penalty
    Uses smooth approximations for differentiability.
    """
    pos = params[:2 * N].reshape(N, 2)
    ang = params[2 * N:]  # degrees

    angles_rad = np.radians(ang)
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + pos[:, 0:1]
    verts_y = ry + pos[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    # Smooth outer side: max over 6 directions of (2/sqrt3)*max|proj|
    # Use a fixed set of candidate phi values (e.g., 36 directions) and smooth max
    phi_candidates = np.radians(np.arange(0, 60, 5))  # 12 directions
    side_components = []
    for phi in phi_candidates:
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        m = abs_proj.max(axis=0)
        side_vals = (2.0 / _SQRT3) * m
        # Smooth max over 6 sides
        beta_side = 10.0
        def logsumexp(arr, beta):
            m = arr.max()
            return m + np.log(np.sum(np.exp((arr - m) / beta))) * beta
        smax = logsumexp(side_vals, beta_side)
        side_components.append(smax)

    side_components = np.array(side_components)
    # Smooth max over phi candidates
    beta_phi = 5.0
    def logsumexp(arr, beta):
        m = arr.max()
        return m + np.log(np.sum(np.exp((arr - m) / beta))) * beta
    outer_side = logsumexp(side_components, beta_phi)

    # Overlap penalty: for each pair, compute minimum separation (negative = overlap)
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq > 4.5:
                continue  # definitely disjoint

            a1 = angles_rad[i]
            a2 = angles_rad[j]
            c1, s1 = np.cos(a1), np.sin(a1)
            c2, s2 = np.cos(a2), np.sin(a2)

            v1x = pos[i, 0] + _UX * c1 - _UY * s1
            v1y = pos[i, 1] + _UX * s1 + _UY * c1
            v2x = pos[j, 0] + _UX * c2 - _UY * s2
            v2y = pos[j, 1] + _UX * s2 + _UY * c2

            en1x = _ENX * c1 - _ENY * s1
            en1y = _ENX * s1 + _ENY * c1
            en2x = _ENX * c2 - _ENY * s2
            en2y = _ENX * s2 + _ENY * c2

            # Compute gaps on all 12 normals
            gaps = []
            for k in range(6):
                nx, ny = en1x[k], en1y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gap1 = p2.min() - p1.max()
                gap2 = p1.min() - p2.max()
                gap = min(0.0, gap1, gap2)
                gaps.append(gap)

            for k in range(6):
                nx, ny = en2x[k], en2y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gap1 = p2.min() - p1.max()
                gap2 = p1.min() - p2.max()
                gap = min(0.0, gap1, gap2)
                gaps.append(gap)

            gaps = np.array(gaps)
            min_gap = np.min(gaps)
            # If min_gap >= 0, no overlap. penalty = softplus(-min_gap)
            softplus_beta = 5.0
            penalty = np.log1p(np.exp(softplus_beta * (-min_gap))) / softplus_beta
            overlap_penalty += penalty

    weight = 50.0
    return outer_side + weight * overlap_penalty


def _cmaes_optimize(x0, sigma0, max_iter, time_limit, seed=None):
    """
    Simple CMA-ES implementation for the 36-dim problem.
    x0: initial center (36,)
    sigma0: initial step size
    max_iter: max generations
    time_limit: seconds
    """
    rng = np.random.RandomState(seed)
    dim = len(x0)
    pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~16 for dim=36
    
    # CMA-ES parameters
    mu = pop_size // 2  # number of parents
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights ** 2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(c1 * 2, (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2) ** 2 + mueff))
    chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    
    mean = x0.copy()
    sigma = sigma0
    C = np.eye(dim)
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    
    best_x = x0.copy()
    best_f = _smooth_objective(x0)
    
    t_start = time.time()
    for gen in range(max_iter):
        if time.time() - t_start > time_limit:
            break
        
        # Generate population
        population = []
        for i in range(pop_size):
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            z = rng.randn(dim)
            x = mean + sigma * L @ z
            population.append(x)
        
        # Evaluate
        fitnesses = np.array([_smooth_objective(x) for x in population])
        
        # Select best mu
        order = np.argsort(fitnesses)
        selected = [population[order[i]] for i in range(mu)]
        selected_f = fitnesses[order[:mu]]
        
        # Update best
        if selected_f[0] < best_f:
            best_f = selected_f[0]
            best_x = selected[0].copy()
        
        # Compute new mean
        new_mean = np.sum([weights[i] * selected[i] for i in range(mu)], axis=0)
        
        # Update evolution paths
        inv_sqrt_C = np.linalg.inv(np.sqrtm(C)) if np.all(np.isfinite(np.sqrtm(C))) else np.eye(dim)
        diff = (new_mean - mean) / sigma
        pc = (1 - cc) * pc + cc * np.sqrt(mueff) * diff
        norm_pc = np.linalg.norm(pc)
        hsig = 1.0 if norm_pc / np.sqrt(1 - (1 - cc) ** (gen + 1)) / np.sqrt(1 - (1 - cc) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi else 0.0
        ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C @ diff
        hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1)) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi else 0.0
        
        # Update covariance
        rank1 = np.outer(pc, pc)
        rankmu = np.zeros((dim, dim))
        for i in range(mu):
            zi = inv_sqrt_C @ (selected[i] - mean) / sigma
            rankmu += weights[i] * np.outer(zi, zi)
        C = (1 - cs * (2 - cs) - c1 - cmu) * C + cs * (2 - cs) * rank1 + c1 * rank1 + cmu * rankmu
        
        # Update sigma
        sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
        
        mean = new_mean
        
        # Ensure sigma doesn't explode
        sigma = np.clip(sigma, 1e-8, 10.0)
        
        # Regularize C
        C = (C + C.T) / 2
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-10, None)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    return best_x, best_f


def _local_polish(pos, ang, time_limit, initial_step=0.03):
    """Coordinate descent polish."""
    current_side, _, _, _ = _compute_min_outer_side(pos, ang)
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang.copy()

    t_start = time.time()
    step = initial_step
    min_step = 0.0005

    while time.time() - t_start < time_limit and step > min_step:
        improved = False
        for idx in range(N):
            if time.time() - t_start > time_limit:
                break
            for dim in range(3):
                if time.time() - t_start > time_limit:
                    break
                for direction in [1, -1]:
                    new_pos = best_pos.copy()
                    new_ang = best_ang.copy()

                    if dim == 0:
                        new_pos[idx, 0] += direction * step
                    elif dim == 1:
                        new_pos[idx, 1] += direction * step
                    else:
                        new_ang[idx] += direction * step

                    angles_rad_new = np.radians(new_ang)
                    valid = True
                    h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                    for j in range(N):
                        if j == idx:
                            continue
                        h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                        if not _hexagons_disjoint_fast(h_moved, h_other):
                            valid = False
                            break

                    if not valid:
                        continue

                    new_side, _, _, _ = _compute_min_outer_side(new_pos, new_ang)

                    if new_side < best_side - 1e-12:
                        best_pos = new_pos
                        best_ang = new_ang
                        best_side = new_side
                        improved = True
                        break

                if time.time() - t_start > time_limit:
                    break
            if time.time() - t_start > time_limit:
                break
            if improved:
                break
        else:
            step *= 0.7
            continue
        break

    return best_side, best_pos, best_ang


def construct():
    t0 = time.time()
    
    # Generate diverse initial configurations
    a1 = np.array([_SQRT3, 0.0])
    a2 = np.array([_SQRT3 / 2.0, 1.5])

    def make_honeycomb():
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
            ang[idx] = 0.0
        return pos, ang

    def make_ring():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i, 0] = 2.5 * _SQRT3 * np.cos(theta)
            pos[i, 1] = 2.5 * _SQRT3 * np.sin(theta)
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            pos[6 + i, 0] = 1.2 * _SQRT3 * np.cos(theta)
            pos[6 + i, 1] = 1.2 * _SQRT3 * np.sin(theta)
        return pos, ang

    def make_two_rows():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i, 0] = i * _SQRT3
            pos[i, 1] = 0.0
        for i in range(6):
            pos[6 + i, 0] = i * _SQRT3 + _SQRT3 / 2
            pos[6 + i, 1] = 1.5
        return pos, ang

    def make_random():
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(-30, 30, N)
        return pos, ang

    # Try multiple initializations with CMA-ES
    inits = [
        make_honeycomb(),
        make_ring(),
        make_two_rows(),
        make_random(),
        make_random(),
        make_random(),
    ]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    n_inits = len(inits)
    for init_idx, (pos0, ang0) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 15.0:
            break

        # Check initial validity
        if not _is_valid(pos0, ang0):
            for _ in range(50):
                if time.time() - t0 > DEADLINE - 15.0:
                    break
                pos0 = pos0 + np.random.uniform(-0.1, 0.1, (N, 2))
                if _is_valid(pos0, ang0):
                    break

        if not _is_valid(pos0, ang0):
            continue

        # Evaluate initial
        s0, phi0, cx0, cy0 = _compute_min_outer_side(pos0, ang0)
        if s0 < best_side:
            best_side = s0
            best_positions = pos0.copy()
            best_angles = ang0.copy()
            best_phi = phi0
            best_cx = cx0
            best_cy = cy0

        # CMA-ES optimization
        params0 = np.zeros(3 * N)
        for i in range(N):
            params0[3 * i] = pos0[i, 0]
            params0[3 * i + 1] = pos0[i, 1]
            params0[3 * i + 2] = ang0[i]

        # Allocate time for CMA-ES
        remaining_inits = n_inits - init_idx
        time_per_init = min(20.0, (DEADLINE - 20.0) / max(1, remaining_inits))
        
        try:
            x_best, f_best = _cmaes_optimize(params0, 0.5, max_iter=50, time_limit=time_per_init, seed=init_idx * 1000 + 42)
            
            if time.time() - t0 > DEADLINE - 10.0:
                break
                
            new_pos = x_best[:2 * N].reshape(N, 2)
            new_ang = x_best[2 * N:]

            if _is_valid(new_pos, new_ang):
                new_side, new_phi, new_cx, new_cy = _compute_min_outer_side(new_pos, new_ang)
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_pos
                    best_angles = new_ang
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
        except Exception:
            pass

    # Local polish on best
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0 and best_positions is not None:
        cd_time = min(remaining_time * 0.5, 15.0)
        cd_side, cd_pos, cd_ang = _local_polish(
            best_positions, best_angles, cd_time, initial_step=0.03)

        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if best_positions is None or not _is_valid(best_positions, best_angles):
        pos0, ang0 = make_honeycomb()
        best_positions = pos0.copy()
        best_angles = ang0.copy()
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
