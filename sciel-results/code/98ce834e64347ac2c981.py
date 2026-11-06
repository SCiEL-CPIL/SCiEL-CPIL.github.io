import numpy as np
import time

# Global time tracking
_TOTAL_START = None
_TOTAL_BUDGET = 315.0  # Safety margin below 330s

def _ensure_time_tracking():
    global _TOTAL_START
    if _TOTAL_START is None:
        _TOTAL_START = time.time()

def _get_global_remaining():
    _ensure_time_tracking()
    return max(0.0, _TOTAL_BUDGET - (time.time() - _TOTAL_START))

def _morse_energy(pos, rho):
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)

def _morse_gradient(pos, rho):
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    exp_val = np.exp(rho * (1.0 - dists))
    pair_force = -2.0 * rho * exp_val * (exp_val - 1.0)
    np.fill_diagonal(pair_force, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pair_force / dists
    W = np.where(dists > 1e-10, W, 0.0)
    Ws = W.sum(axis=1)
    grad = Ws[:, None] * pos - W @ pos
    return grad

def _min_dist_check(pos, min_d=0.5):
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9

def _ensure_min_dist(pos, min_d=0.51, max_iter=5000):
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if not _min_dist_check(pos, min_d):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
            rij = pos[idx_i] - pos[idx_j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                rij = np.array([1.0, 0.0, 0.0])
                r = 1.0
            push = (min_d - r + 0.02) / 2.0
            pos[idx_i] += push * rij / r
            pos[idx_j] -= push * rij / r
        else:
            break
    return pos

def _fcc_points(n, spacing=1.0):
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1 / 3))) + 10
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _hcp_points(n, spacing=1.0):
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1 / 3))) + 10
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1.0 / 3.0, j + 2.0 / 3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _bcp_points(n, spacing=1.0):
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1 / 3))) + 10
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _icosahedral_points(n, spacing=1.0):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v.copy())

    # Generate shell 2 (42 atoms)
    edge_midpoints = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2.0
                r = np.linalg.norm(mid)
                if r > 1e-10:
                    mid = mid / r * 2.0
                    edge_midpoints.append(mid)
    
    faces = [
        [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
        [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
        [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
        [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
        [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
    ]
    face_centers = []
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        r = np.linalg.norm(center)
        if r > 1e-10:
            center = center / r * 2.0
            face_centers.append(center)
    
    for m in edge_midpoints:
        pts.append(m.copy())
    for f in face_centers:
        pts.append(f.copy())
    
    if len(pts) < n:
        for v in verts:
            pts.append(v * 3.0)
        for m in edge_midpoints:
            pts.append(m * 1.5)
        for f in face_centers:
            pts.append(f * 1.5)
    
    rng_local = np.random.default_rng(123)
    while len(pts) < n:
        p = pts[rng_local.integers(len(pts))] + rng_local.normal(0, 0.15, 3)
        pts.append(p)
    
    pts = np.array(pts[:n])
    
    if len(pts) > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d > 1e-10:
            pts = pts * (spacing / min_d)
    
    return pts

def _random_dense(n, rng, spacing=1.0):
    pts = []
    radius = spacing * (n ** (1 / 3)) * 1.5
    min_d = 0.53 * spacing
    max_attempts = 200000
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    valid = False
                    break
            if valid:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)

def _relax(pos, rho, maxiter=2000, tol=1e-12):
    from scipy.optimize import minimize
    n = len(pos)
    x0 = pos.flatten()

    def objective(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)

    def gradient(x):
        p = x.reshape(n, 3)
        return _morse_gradient(p, rho).flatten()

    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': tol, 'ftol': 1e-14})
        e = float(result.fun)
        if e < _morse_energy(pos, rho) + 1e-9:
            return result.x.reshape(n, 3), e
    except Exception:
        pass
    return pos, _morse_energy(pos, rho)

def _relax_fast(pos, rho, maxiter=300):
    from scipy.optimize import minimize
    n = len(pos)
    x0 = pos.flatten()

    def objective(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)

    def gradient(x):
        p = x.reshape(n, 3)
        return _morse_gradient(p, rho).flatten()

    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10})
        e = float(result.fun)
        if e < _morse_energy(pos, rho) + 1e-9:
            return result.x.reshape(n, 3), e
    except Exception:
        pass
    return pos, _morse_energy(pos, rho)

def _sa_basin_hopping(pos, rho, deadline, rng):
    """
    Perform Simulated Annealing with Basin Hopping.
    """
    current_pos = pos.copy()
    current_energy = _morse_energy(pos, rho)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Annealing schedule
    T_start = 0.5
    T_end = 0.01
    n_steps = 200
    alpha = (T_end / T_start) ** (1.0 / n_steps)
    T = T_start
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Kick
        kick_size = 0.1 + 0.2 * (T / T_start)
        kicked = current_pos + rng.normal(0, kick_size, current_pos.shape)
        kicked = _ensure_min_dist(kicked, min_d=0.51)
        
        if not _min_dist_check(kicked):
            T *= alpha
            continue
        
        relaxed, energy = _relax_fast(kicked, rho, maxiter=300)
        
        if _min_dist_check(relaxed):
            if energy < current_energy:
                # Always accept downhill moves
                current_pos = relaxed.copy()
                current_energy = energy
            else:
                # Metropolis acceptance for uphill moves
                delta_e = energy - current_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                    current_pos = relaxed.copy()
                    current_energy = energy
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
        
        T *= alpha
    
    return best_pos, best_energy

def construct(instance):
    global _TOTAL_START
    _ensure_time_tracking()

    size = instance["n"]
    rho = instance.get("rho", 10)

    if _get_global_remaining() < 10.0:
        return _random_dense(size, np.random.default_rng(0))

    instance_start = time.time()
    instance_budget = min(48.0, _get_global_remaining() - 8.0)
    deadline = instance_start + instance_budget

    rng = np.random.default_rng(size * 17 + int(rho) * 23 + 7)

    best_energy = np.inf
    best_pos = None

    # --- Phase 1: Generate diverse structural seeds ---
    seeds = []

    if rho <= 4:
        # Long range: FCC and HCP dominate
        seeds.append(_fcc_points(size))
        seeds.append(_hcp_points(size))
        seeds.append(_bcp_points(size))
        seeds.append(_random_dense(size, rng))
        seeds.append(_random_dense(size, rng))
    elif rho <= 8:
        # Medium range: mix of icosahedral and close-packed
        seeds.append(_icosahedral_points(size))
        seeds.append(_fcc_points(size))
        seeds.append(_hcp_points(size))
        seeds.append(_bcp_points(size))
        seeds.append(_random_dense(size, rng))
        seeds.append(_random_dense(size, rng))
    else:
        # Short range: icosahedral shells dominate
        seeds.append(_icosahedral_points(size))
        seeds.append(_fcc_points(size))
        seeds.append(_random_dense(size, rng))
        seeds.append(_random_dense(size, rng))
        seeds.append(_random_dense(size, rng))

    # Add perturbed versions of top seeds
    for base in seeds[:4]:
        for sigma in [0.08, 0.15, 0.25]:
            seeds.append(base + rng.normal(0, sigma, base.shape))

    # --- Phase 2: Relax all seeds ---
    for pos in seeds:
        if time.time() > deadline - 10.0:
            break
        pos = _ensure_min_dist(pos)
        if not _min_dist_check(pos):
            continue
        relaxed_pos, energy = _relax(pos, rho, maxiter=1500)
        if _min_dist_check(relaxed_pos):
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()

    if best_pos is None:
        pos = _random_dense(size, rng)
        pos = _ensure_min_dist(pos)
        best_pos, best_energy = _relax(pos, rho, maxiter=2000)

    # --- Phase 3: Simulated Annealing Basin Hopping ---
    remaining = deadline - time.time()
    if remaining > 10.0:
        # Use SA basin hopping for the remaining time
        sa_pos, sa_energy = _sa_basin_hopping(best_pos, rho, deadline, rng)
        if _min_dist_check(sa_pos) and sa_energy < best_energy:
            best_energy = sa_energy
            best_pos = sa_pos.copy()

    # --- Phase 4: Final high-precision polish ---
    if best_pos is not None and time.time() < deadline - 3.0:
        relaxed_pos, energy = _relax(best_pos, rho, maxiter=5000, tol=1e-14)
        if _min_dist_check(relaxed_pos) and energy < best_energy:
            best_pos = relaxed_pos
            best_energy = energy

    # Final safety: ensure minimum distance
    best_pos = _ensure_min_dist(best_pos, min_d=0.51)

    return best_pos
