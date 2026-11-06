import numpy as np
import time
from scipy.optimize import minimize

# Global time tracking
_TOTAL_START = None
_TOTAL_BUDGET = 310.0  # Safety margin below 330s

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
    # dU/dr = -2*rho*exp(rho*(1-r))*(exp(rho*(1-r))-1)
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
    max_r = int(np.ceil(n ** (1 / 3))) + 5
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
    max_r = int(np.ceil(n ** (1 / 3))) + 5
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

def _icosahedral_points(n, spacing=1.0):
    """
    Generate points based on Icosahedral symmetry.
    For N=38, 50, 80, these are not perfect Mackay shells, so we generate
    a dense icosahedral-like cloud and let the optimizer shape it.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    for v in verts:
        pts.append(v.copy())
        
    # Shell 2: 30 edge midpoints
    v_norm = verts
    edge_midpoints = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(v_norm[i] - v_norm[j])
            if 0.9 < d < 1.1:
                mid = (v_norm[i] + v_norm[j]) / 2.0
                r = np.linalg.norm(mid)
                if r > 1e-10:
                    mid = mid / r * 1.155
                    edge_midpoints.append(mid)
    
    unique_mids = []
    for m in edge_midpoints:
        if not any(np.linalg.norm(m - u) < 0.01 for u in unique_mids):
            unique_mids.append(m)
            
    for m in unique_mids:
        pts.append(m.copy())
        
    # Shell 3: 20 face centers
    faces = [
        [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
        [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
        [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
        [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
        [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
    ]
    for f in faces:
        center = (v_norm[f[0]] + v_norm[f[1]] + v_norm[f[2]]) / 3.0
        r = np.linalg.norm(center)
        if r > 1e-10:
            center = center / r * 1.323
            pts.append(center)
            
    # If we need more points (e.g., N=80), add a second icosahedral shell
    if n > len(pts):
        # Scale existing outer shell
        verts2 = v_norm * 2.0
        for v in verts2:
            pts.append(v.copy())
        for m in unique_mids:
            pts.append(m * 2.0)
        for f in faces:
            center = (v_norm[f[0]] + v_norm[f[1]] + v_norm[f[2]]) / 3.0
            r = np.linalg.norm(center)
            if r > 1e-10:
                center = center / r * 1.323 * 2.0
                pts.append(center)
                
    # If still not enough, add random points in the shell region
    rng = np.random.default_rng(42)
    while len(pts) < n:
        p = pts[rng.integers(len(pts))] + rng.normal(0, 0.1, 3)
        pts.append(p)
        
    pts = np.array(pts[:n])
    
    # Scale to unit nearest-neighbor distance
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
    radius = spacing * (n ** (1 / 3)) * 1.2
    min_d = 0.52 * spacing
    max_attempts = 100000
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

def _relax(pos, rho, maxiter=1000, tol=1e-10):
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
                          options={'maxiter': maxiter, 'gtol': tol, 'ftol': 1e-12})
        if result.fun < _morse_energy(pos, rho) + 1e-9:
            return result.x.reshape(n, 3), float(result.fun)
    except Exception:
        pass
    return pos, _morse_energy(pos, rho)

def construct(instance):
    global _TOTAL_START
    _ensure_time_tracking()
    
    size = instance["n"]
    rho = instance.get("rho", 10)
    
    # Check global time
    if _get_global_remaining() < 5.0:
        # Emergency fallback
        return _random_dense(size, np.random.default_rng(0))
        
    instance_start = time.time()
    instance_budget = min(45.0, _get_global_remaining() - 5.0)
    deadline = instance_start + instance_budget
    
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # 1. Generate diverse seeds
    seeds = []
    if rho <= 4:
        seeds.append(_fcc_points(size))
        seeds.append(_hcp_points(size))
        seeds.append(_random_dense(size, rng))
        seeds.append(_random_dense(size, rng))
    elif rho <= 8:
        seeds.append(_fcc_points(size))
        seeds.append(_hcp_points(size))
        seeds.append(_icosahedral_points(size))
        seeds.append(_random_dense(size, rng))
    else:
        seeds.append(_icosahedral_points(size))
        seeds.append(_fcc_points(size))
        seeds.append(_random_dense(size, rng))
        seeds.append(_random_dense(size, rng))
        
    # Add perturbed seeds
    for base in seeds[:3]:
        for sigma in [0.1, 0.2]:
            seeds.append(base + rng.normal(0, sigma, base.shape))
            
    # 2. Relax seeds
    for pos in seeds:
        if time.time() > deadline - 5.0:
            break
        pos = _ensure_min_dist(pos)
        if not _min_dist_check(pos):
            continue
        relaxed_pos, energy = _relax(pos, rho, maxiter=1000)
        if _min_dist_check(relaxed_pos):
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
                
    # 3. Basin Hopping / Kick-and-Relax
    if best_pos is None:
        # Fallback
        pos = _random_dense(size, rng)
        pos = _ensure_min_dist(pos)
        best_pos, best_energy = _relax(pos, rho, maxiter=2000)
        
    # Adaptive kick sizes
    kick_sizes = [0.3, 0.2, 0.15, 0.1, 0.05, 0.02]
    # We have ~30s left. Each relaxation ~50ms. 30s / 0.05s = 600 kicks.
    total_kicks_budget = int((deadline - time.time()) * 15) # ~15 kicks per second
    kicks_per_size = max(1, total_kicks_budget // len(kick_sizes))
    
    for kick_size in kick_sizes:
        for _ in range(kicks_per_size):
            if time.time() > deadline - 5.0:
                break
            kicked_pos = best_pos + rng.normal(0, kick_size, best_pos.shape)
            kicked_pos = _ensure_min_dist(kicked_pos)
            if not _min_dist_check(kicked_pos):
                continue
            relaxed_pos, energy = _relax(kicked_pos, rho, maxiter=500) # Faster relax for kicks
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
                    
    # 4. Final Polish
    if best_pos is not None and time.time() < deadline - 2.0:
        relaxed_pos, energy = _relax(best_pos, rho, maxiter=5000, tol=1e-12)
        if _min_dist_check(relaxed_pos) and energy < best_energy:
            best_pos = relaxed_pos
            best_energy = energy
            
    return best_pos
