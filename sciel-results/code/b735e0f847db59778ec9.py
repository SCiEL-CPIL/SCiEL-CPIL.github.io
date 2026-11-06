"""Morse cluster global minima finder - Enhanced Structural & Stochastic Search.

Strategy:
1. Generate high-quality initial configurations based on rho (FCC/HCP for long range, Icosahedral for short range).
2. Use robust L-BFGS-B optimization with analytic gradients to relax each configuration.
3. Perform stochastic perturbations (kick-and-relax) to escape local minima.
4. Strict time management to ensure all 6 instances finish within 300s.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

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
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    # Force magnitude per pair: dE/dr = -2 * rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1)
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

def _ensure_min_dist(pos, min_d=0.51, max_iter=500):
    """Separate atoms that are too close using a simple repulsion scheme."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if not _min_dist_check(pos, min_d):
            # Find the closest pair
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
        else:
            break
    return pos

def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
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
    max_r = int(np.ceil(n ** (1/3))) + 4
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

def _mckay_icosahedron(n, spacing=1.0):
    """Generate Mackay icosahedron cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Standard Mackay icosahedron has shells of 1, 12, 42, 92...
    # We will generate points on icosahedral shells
    
    pts = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    for v in verts:
        pts.append(v.copy())
    
    # Shell 2: 42 points (centers of faces and edges)
    faces = [
        [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
        [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
        [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
        [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
        [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
    ]
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        # Normalize to shell radius ~ 1.15 * edge_length
        r = np.linalg.norm(center)
        if r > 1e-10:
            center = center / r * 1.15
            pts.append(center)
            
    # Shell 3: 92 points (approx)
    # For simplicity, we will just perturb the existing points and add more if needed
    # A better way is to generate icosahedral lattice points
    
    # If we need more points, we scale up and add
    if n > len(pts):
        # Add second shell of vertices
        verts2 = verts * 2.0
        for v in verts2:
            pts.append(v.copy())
        # Add second shell of face centers
        for f in faces:
            center = (verts2[f[0]] + verts2[f[1]] + verts2[f[2]]) / 3.0
            pts.append(center)
            
    # If still not enough, just randomize
    while len(pts) < n:
        p = pts[np.random.randint(len(pts))] + np.random.normal(0, 0.1, 3)
        pts.append(p)
        
    pts = np.array(pts[:n])
    # Scale to unit edge length roughly
    # Find min distance
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
    radius = spacing * (n ** (1/3)) * 0.8
    min_d = 0.55 * spacing
    max_attempts = 50000
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

def _basin_hopping(pos, rho, n_iter=50, kick_scale=0.5, max_iter_relax=2000, time_limit=30.0):
    """Perform Basin Hopping optimization."""
    start_time = time.time()
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Kick
        kick = rng.normal(0, kick_scale, current_pos.shape)
        new_pos = current_pos + kick
        new_pos = _ensure_min_dist(new_pos)
        
        # 2. Relax
        try:
            x0 = new_pos.flatten()
            def objective(x):
                p = x.reshape(-1, 3)
                return _morse_energy(p, rho)
            def gradient(x):
                p = x.reshape(-1, 3)
                return _morse_gradient(p, rho).flatten()
                
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': max_iter_relax, 'gtol': 1e-10, 'ftol': 1e-12})
            
            relaxed_pos = result.x.reshape(-1, 3)
            relaxed_energy = result.fun
            
            if not _min_dist_check(relaxed_pos):
                continue
                
            # 3. Accept/Reject (Metropolis or just keep if better)
            # For global min search, we usually accept if better, or use Metropolis.
            # Since we want the global min, we just track the best.
            if relaxed_energy < best_energy:
                best_energy = relaxed_energy
                best_pos = relaxed_pos.copy()
                
            # Update current position
            # Simple: always move to new local min if valid
            current_pos = relaxed_pos
            current_energy = relaxed_energy
            
        except Exception as e:
            continue
            
    return best_pos, best_energy

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    best_energy = np.inf
    best_pos = None
    
    # Generate initial configurations based on rho
    initial_configs = []
    
    # Determine structural bias
    if rho <= 3:
        # Long range: FCC/HCP
        initial_configs.append(('fcc', _fcc_points(size)))
        initial_configs.append(('hcp', _hcp_points(size)))
    elif rho <= 6:
        # Medium range: Mix
        initial_configs.append(('fcc', _fcc_points(size)))
        initial_configs.append(('hcp', _hcp_points(size)))
        initial_configs.append(('ico', _mckay_icosahedron(size)))
    else:
        # Short range: Icosahedral
        initial_configs.append(('ico', _mckay_icosahedron(size)))
        initial_configs.append(('fcc', _fcc_points(size)))
        
    # Add random restarts
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    for i in range(5):
        initial_configs.append((f'rand_{i}', _random_dense(size, rng)))
        
    # Run Basin Hopping on each
    for name, pos in initial_configs:
        if time.time() > deadline - 10:
            break
            
        try:
            # Adjust kick scale and iterations based on time remaining
            time_left = deadline - time.time()
            n_iter = max(10, int(time_left / 5)) # ~5s per iteration estimate
            
            relaxed_pos, energy = _basin_hopping(pos, rho, n_iter=n_iter, kick_scale=0.5, time_limit=time_left - 10)
            
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception as e:
            continue
            
    # If no valid result, fallback
    if best_pos is None:
        best_pos = _mckay_icosahedron(size)
        best_pos = _ensure_min_dist(best_pos)
        
    return best_pos
# EVOLVE-BLOCK-END
