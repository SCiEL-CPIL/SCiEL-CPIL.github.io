"""Morse cluster global minima finder - Multi-start with diverse structural seeds and basin-hopping."""
import numpy as np
import time
from scipy.optimize import minimize, basinhopping

_TOTAL_START = None
_INSTANCE_BUDGET = 45.0


def _ensure_time_tracking():
    global _TOTAL_START
    if _TOTAL_START is None:
        _TOTAL_START = time.time()


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


def _ensure_min_dist(pos, min_d=0.51, max_iter=500):
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
            push = (min_d - r + 0.01) / 2.0
            pos[idx_i] += push * rij / r
            pos[idx_j] -= push * rij / r
        else:
            break
    return pos


def _fcc_points(n, spacing=1.0):
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1 / 3))) + 6
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
    max_r = int(np.ceil(n ** (1 / 3))) + 6
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


def _bcc_points(n, spacing=1.0):
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1 / 3))) + 6
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


def _decahedron_points(n, spacing=1.0):
    """Generate decahedral (Mackay-like) cluster."""
    pts = [np.zeros(3)]
    # First shell: 12 vertices of a decahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    for v in verts:
        pts.append(v.copy() * 1.0)
    
    if n > 13:
        # Second shell: face centers
        faces = [
            [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
            [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
            [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
            [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
            [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
        ]
        for f in faces:
            center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            r = np.linalg.norm(center)
            if r > 1e-10:
                center = center / r * 1.323
                pts.append(center.copy())
        
        if n > 33:
            # Third shell: edge midpoints
            for i in range(12):
                for j in range(i + 1, 12):
                    d = np.linalg.norm(verts[i] - verts[j])
                    if 0.9 < d < 1.1:
                        mid = (verts[i] + verts[j]) / 2.0
                        r = np.linalg.norm(mid)
                        if r > 1e-10:
                            mid = mid / r * 1.155
                            pts.append(mid.copy())
    
    # Pad with random points if needed
    rng = np.random.default_rng(42)
    while len(pts) < n:
        p = pts[rng.integers(len(pts))] + rng.normal(0, 0.05, 3)
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
    radius = spacing * (n ** (1 / 3)) * 0.9
    min_d = 0.52 * spacing
    max_attempts = 30000
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


def _relax_lbfgs(pos, rho, maxiter=2000):
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
                          options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50})
        if result.fun < _morse_energy(pos, rho) + 1e-12:
            return result.x.reshape(n, 3), float(result.fun)
    except Exception:
        pass
    return pos, _morse_energy(pos, rho)


def construct(instance):
    _ensure_time_tracking()
    instance_start = time.time()
    
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    deadline = instance_start + _INSTANCE_BUDGET
    
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    if rho <= 4:
        initial_configs.append(_fcc_points(size))
        initial_configs.append(_hcp_points(size))
        initial_configs.append(_bcc_points(size))
        initial_configs.append(_decahedron_points(size))
        initial_configs.append(_random_dense(size, rng))
    elif rho <= 8:
        initial_configs.append(_fcc_points(size))
        initial_configs.append(_hcp_points(size))
        initial_configs.append(_decahedron_points(size))
        initial_configs.append(_bcc_points(size))
        initial_configs.append(_random_dense(size, rng))
    else:
        initial_configs.append(_decahedron_points(size))
        initial_configs.append(_fcc_points(size))
        initial_configs.append(_bcc_points(size))
        initial_configs.append(_random_dense(size, rng))
    
    # Relax each initial configuration
    best_energy = np.inf
    best_pos = None
    
    for pos in initial_configs:
        if time.time() > deadline - 5.0:
            break
        pos = _ensure_min_dist(pos)
        if not _min_dist_check(pos):
            continue
        try:
            relaxed_pos, energy = _relax_lbfgs(pos, rho, maxiter=2000)
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    # Basin-hopping search from best configuration
    if best_pos is not None and time.time() < deadline - 10.0:
        n = size
        x0 = best_pos.flatten()
        
        def bh_objective(x):
            p = x.reshape(n, 3)
            return _morse_energy(p, rho)
        
        def bh_gradient(x):
            p = x.reshape(n, 3)
            return _morse_gradient(p, rho).flatten()
        
        for kick_scale in [0.3, 0.2, 0.15, 0.1]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                result = basinhopping(
                    bh_objective, x0,
                    minimizer_kwargs={'jac': bh_gradient, 'method': 'L-BFGS-B',
                                      'options': {'maxiter': 1000, 'gtol': 1e-10}},
                    stepsize=kick_scale,
                    niter=25,
                    seed=int(rho) * 100 + size,
                    accept_test=lambda f_new, f_old: f_new < f_old,
                    interval=50,
                    disp=False
                )
                if result.fun < best_energy:
                    best_energy = float(result.fun)
                    best_pos = result.x.reshape(n, 3).copy()
                    if not _min_dist_check(best_pos):
                        best_pos = _ensure_min_dist(best_pos)
            except Exception:
                pass
    
    # Final kick-and-relax refinement
    if best_pos is not None and time.time() < deadline - 5.0:
        for kick_size in [0.1, 0.08, 0.05, 0.03]:
            if time.time() > deadline - 5.0:
                break
            for _ in range(4):
                if time.time() > deadline - 5.0:
                    break
                kicked_pos = best_pos + rng.normal(0, kick_size, best_pos.shape)
                kicked_pos = _ensure_min_dist(kicked_pos)
                if not _min_dist_check(kicked_pos):
                    continue
                try:
                    relaxed_pos, energy = _relax_lbfgs(kicked_pos, rho, maxiter=1000)
                    if _min_dist_check(relaxed_pos):
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = relaxed_pos.copy()
                except Exception:
                    continue
    
    # Final polish
    if best_pos is not None and time.time() < deadline - 2.0:
        try:
            relaxed_pos, energy = _relax_lbfgs(best_pos, rho, maxiter=3000)
            if _min_dist_check(relaxed_pos) and energy < best_energy:
                best_pos = relaxed_pos
                best_energy = energy
        except Exception:
            pass
    
    if best_pos is None:
        best_pos = _random_dense(size, rng)
        best_pos = _ensure_min_dist(best_pos)
        
    return best_pos
