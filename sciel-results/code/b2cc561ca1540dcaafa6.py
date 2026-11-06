"""Morse cluster global minima finder - Enhanced with BCC, multiple structures, and simulated annealing.

Strategy:
1. Generate diverse initial configurations: FCC, HCP, BCC, Icosahedral, Random.
2. Use L-BFGS-B with analytic gradients for fast local relaxation.
3. Simulated annealing with multiple restarts to escape local minima.
4. Strict time management to ensure all 6 instances finish within 300s.
"""
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
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
    """Compute gradient of Morse energy (fully vectorized)."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
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
    """Check minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9

def _ensure_min_dist(pos, min_d=0.51, max_iter=1000):
    """Separate atoms that are too close using a simple repulsion scheme."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if _min_dist_check(pos, min_d):
            break
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
        if idx_i > idx_j:
            idx_i, idx_j = idx_j, idx_i
        rij = pos[idx_i] - pos[idx_j]
        r = np.linalg.norm(rij)
        if r < 1e-10:
            rij = np.array([1.0, 0, 0])
            r = 1.0
        push = (min_d - r + 0.01) / 2.0
        pos[idx_i] += push * rij / r
        pos[idx_j] -= push * rij / r
    return pos

def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 5
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
    """Generate HCP lattice points."""
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 5
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 5
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

def _icosahedral_cluster(n, spacing=1.0):
    """Generate icosahedral cluster points."""
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
    
    # Add face centers for more points
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
            center = center / r * 1.15
            pts.append(center)
    
    # Add second shell if needed
    if n > len(pts):
        verts2 = verts * 2.0
        for v in verts2:
            pts.append(v.copy())
        for f in faces:
            center = (verts2[f[0]] + verts2[f[1]] + verts2[f[2]]) / 3.0
            pts.append(center)
    
    while len(pts) < n:
        p = pts[np.random.randint(len(pts))] + np.random.normal(0, 0.1, 3)
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
    """Generate random dense packing."""
    pts = []
    radius = spacing * (n ** (1/3)) * 0.9
    min_d = 0.55 * spacing
    max_attempts = 20000
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

def _relax_lbfgs(pos, rho, maxiter=1500):
    """Relax using L-BFGS-B."""
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
                          options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 50})
        if result.fun < _morse_energy(pos, rho):
            return result.x.reshape(n, 3), result.fun
    except Exception:
        pass

    return pos, _morse_energy(pos, rho)

def _simulated_annealing(pos, rho, rng, t_start=1.0, t_end=0.01, n_iter=500, kick_scale=0.3):
    """Simple simulated annealing to escape local minima."""
    n = len(pos)
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for i in range(n_iter):
        t = t_start * (t_end / t_start) ** (i / n_iter)
        kick = rng.normal(0, kick_scale, current_pos.shape)
        new_pos = current_pos + kick
        new_pos = _ensure_min_dist(new_pos)
        if not _min_dist_check(new_pos):
            continue
        
        new_energy = _morse_energy(new_pos, rho)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < np.exp(-delta / t):
            current_pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_pos, best_energy

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 48.0
    
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate initial configurations
    initial_configs = []
    
    if rho <= 6:
        # Long/Medium range: FCC, HCP, BCC, Random
        initial_configs.append(_fcc_points(size))
        initial_configs.append(_hcp_points(size))
        initial_configs.append(_bcc_points(size))
        initial_configs.append(_random_dense(size, rng))
        initial_configs.append(_random_dense(size, rng))
    else:
        # Short range: Icosahedral, FCC, BCC, Random
        initial_configs.append(_icosahedral_cluster(size))
        initial_configs.append(_fcc_points(size))
        initial_configs.append(_bcc_points(size))
        initial_configs.append(_random_dense(size, rng))
        initial_configs.append(_random_dense(size, rng))
    
    # Add perturbed versions of the best structural candidates
    for base in initial_configs[:3]:
        for kick in [0.05, 0.1, 0.15]:
            perturbed = base + rng.normal(0, kick, base.shape)
            initial_configs.append(perturbed)
    
    # Relax each configuration
    for pos in initial_configs:
        if time.time() > deadline - 15.0:
            break
            
        pos = _ensure_min_dist(pos)
        if not _min_dist_check(pos):
            continue
            
        try:
            relaxed_pos, energy = _relax_lbfgs(pos, rho, maxiter=1500)
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception:
            continue
            
    # Simulated annealing from the best configuration
    if best_pos is not None:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 5.0:
            n_sa_iters = min(1000, int(remaining_time * 50))
            try:
                sa_pos, sa_energy = _simulated_annealing(
                    best_pos, rho, rng,
                    t_start=0.5, t_end=0.01,
                    n_iter=n_sa_iters, kick_scale=0.2
                )
                if sa_energy < best_energy:
                    best_energy = sa_energy
                    best_pos = sa_pos.copy()
            except Exception:
                pass
    
    # Additional kick-and-relax cycles if time remains
    if best_pos is not None:
        while time.time() < deadline - 8.0:
            kick_size = 0.15
            kicked_pos = best_pos + rng.normal(0, kick_size, best_pos.shape)
            kicked_pos = _ensure_min_dist(kicked_pos)
            if not _min_dist_check(kicked_pos):
                continue
                
            try:
                relaxed_pos, energy = _relax_lbfgs(kicked_pos, rho, maxiter=800)
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
