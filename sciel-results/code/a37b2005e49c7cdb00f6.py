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
from autoresearch_suite.chem_clusters import morse_energy_gradient, relax


def _morse_energy(pos, rho):
    """Compute Morse potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return float(energy)


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # energy = sum_{i<j} ex*(ex-2)/2
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    # gradient: dE/dpos_i = sum_{j!=i} ex*(ex-1)*rho/dists_ij * (pos_i - pos_j)
    # Actually: d/dr [exp(rho(1-r))*(exp(rho(1-r))-2)] = -rho*exp(rho(1-r))*(exp(rho(1-r))-1)
    # dE/dpos_i = sum_{j!=i} d/dr_ij * (pos_i - pos_j)/r_ij
    # d/dr_ij of [exp(x)*(exp(x)-2)] where x=rho(1-r) is -rho*exp(x)*(exp(x)-1)
    grad_factor = -rho * ex * (ex - 1.0) / dists
    np.fill_diagonal(grad_factor, 0.0)
    # grad[i] = sum_j grad_factor[i,j] * (pos[i] - pos[j])
    # = sum_j grad_factor[i,j] * pos[i] - sum_j grad_factor[i,j] * pos[j]
    # = pos[i] * sum_j grad_factor[i,j] - sum_j grad_factor[i,j] * pos[j]
    row_sum = np.sum(grad_factor, axis=1, keepdims=True)
    grad = row_sum * pos - grad_factor @ pos
    return float(energy), grad


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
        r = np.linalg.norm(center)
        if r > 1e-10:
            center = center / r * 1.15
            pts.append(center)
            
    # Shell 3: 92 points (approx)
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
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d > 1e-10:
            pts = pts * (spacing / min_d)
            
    return pts


def _random_packed_init(n, spacing=1.0, seed=None):
    """Generate random packed initial positions."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _ensure_min_dist(pos, min_d=0.51, max_iter=500):
    """Separate atoms that are too close using a simple repulsion scheme."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if n < 2:
            break
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d_actual = np.min(dists)
        if min_d_actual >= min_d:
            break
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


def _basin_hopping(pos, rho, max_iter=20, step_size=0.5, deadline=None):
    """Perform basin hopping to escape local minima."""
    best_pos = pos.copy()
    best_energy = _morse_energy(pos, rho)
    current_pos = pos.copy()
    current_energy = best_energy
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        perturbation = np.random.normal(0, step_size, current_pos.shape)
        new_pos = current_pos + perturbation
        new_pos = _ensure_min_dist(new_pos, 0.51)
        
        # Local optimization
        try:
            result = minimize(
                _morse_energy_grad,
                new_pos,
                args=(rho,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_energy = result.fun
            new_pos = result.x
        except Exception:
            continue
        
        # Metropolis acceptance
        if new_energy < current_energy or np.random.rand() < np.exp(-(new_energy - current_energy) / 0.1):
            current_pos = new_pos
            current_energy = new_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations using multi-start local optimization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if isinstance(parameter, (int, float)) else 3
    
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    # Generate diverse initial configurations
    inits = []
    
    # Choose lattice type based on rho
    if rho <= 3:
        lattice_types = ['fcc', 'hcp']
    elif rho <= 6:
        lattice_types = ['fcc', 'hcp', 'mckay']
    else:
        lattice_types = ['mckay', 'fcc']
    
    for lattice in lattice_types:
        if lattice == 'fcc':
            pos = _fcc_points(size)
        elif lattice == 'hcp':
            pos = _hcp_points(size)
        else:
            pos = _mckay_icosahedron(size)
        pos = _ensure_min_dist(pos, 0.51)
        inits.append(pos)
    
    # Random packed inits with different seeds
    for seed in [42, 123, 777, 2024]:
        pos = _random_packed_init(size, spacing=1.0, seed=seed)
        pos = _ensure_min_dist(pos, 0.51)
        inits.append(pos)
    
    # Scaled lattice inits (slightly perturbed)
    for lattice in lattice_types[:2]:
        if lattice == 'fcc':
            pos = _fcc_points(size)
        else:
            pos = _hcp_points(size)
        rng = np.random.default_rng(hash(lattice) % 2**32)
        pos += rng.normal(0, 0.1, pos.shape)
        pos = _ensure_min_dist(pos, 0.51)
        inits.append(pos)
    
    best_energy = np.inf
    best_positions = inits[0].copy()
    
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Try local optimization with L-BFGS-B
        try:
            result = minimize(
                _morse_energy_grad,
                init_pos,
                args=(rho,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            energy = result.fun
            if energy < best_energy:
                best_energy = energy
                best_positions = result.x.copy()
        except Exception:
            pass
        
        # Also try the provided relax function
        if time.time() > deadline:
            break
        try:
            energy2, pos2 = relax(init_pos.copy(), potential, parameter)
            if energy2 < best_energy:
                best_energy = energy2
                best_positions = pos2.copy()
        except Exception:
            pass
        
        # Basin hopping for the best configuration so far
        if time.time() > deadline:
            break
        try:
            bh_pos, bh_energy = _basin_hopping(best_positions, rho, max_iter=15, step_size=0.4, deadline=deadline)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
        except Exception:
            pass
    
    # Final check: ensure minimum distance constraint
    best_positions = _ensure_min_dist(best_positions, 0.51)
    
    return best_positions
# EVOLVE-BLOCK-END
