"""
Morse cluster global minima via Deterministic Multi-Start L-BFGS-B with diverse structural seeds.
Strategy:
1. Generate a rich library of structurally diverse seeds (FCC, HCP, BCC, Icosahedral, Random).
2. Pre-relax each seed with L-BFGS-B (scipy.optimize.minimize) to a local minimum.
3. Run additional L-BFGS-B starts from perturbed versions of the best seeds.
4. Return the global best found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
_DEADLINE = _START_TIME + 285.0

def _time_remaining():
    return _DEADLINE - time.time()

def _morse_energy(coords, rho):
    """Compute Morse energy; return inf if any pair < 0.5 apart."""
    n = len(coords)
    if n < 2:
        return 0.0
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        return np.inf
    iu = np.triu_indices(n, 1)
    r = dists[iu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)

def _morse_energy_gradient(coords, rho):
    """Compute Morse energy and gradient."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros((n, 3))
    
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    if np.min(dists) < 0.5:
        return np.inf, np.zeros((n, 3))
    
    iu = np.triu_indices(n, 1)
    r = dists[iu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    
    # Gradient: dE/dr_ij * dr_ij/dx_i
    # dE/dr = -rho * e * (2e - 1)
    dE_dr = -rho * e * (2.0 * e - 1.0)
    
    grad = np.zeros((n, 3))
    for i, j in zip(iu[0], iu[1]):
        # dr/dx_i = (x_i - x_j) / r
        # dr/dx_j = (x_j - x_i) / r
        drij_dxi = diff[i, j] / r[j]  # (x_i - x_j) / r
        drij_dxj = -drij_dxi
        
        grad[i] += dE_dr[j] * drij_dxi
        grad[j] += dE_dr[j] * drij_dxj
    
    return energy, grad

def _local_minimize(coords, rho, maxiter=100):
    """Run L-BFGS-B minimization from initial coordinates."""
    def fun(x):
        c = x.reshape(-1, 3)
        e, g = _morse_energy_gradient(c, rho)
        if not np.isfinite(e):
            return e, np.zeros_like(x)
        return e, g.reshape(-1)
    
    x0 = coords.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-6})
        if res.success or res.fun < 1e10:
            return res.fun, res.x.reshape(-1, 3)
    except:
        pass
    
    # Fallback to relax
    try:
        energy, relaxed = relax(coords, 'morse', rho)
        if np.isfinite(energy):
            return energy, relaxed
    except:
        pass
    
    return np.inf, coords

def _generate_fcc_seeds(n, rng):
    """Generate FCC lattice fragments."""
    seeds = []
    a = 1.0
    max_idx = int(np.ceil((n / 4) ** (1/3))) + 2
    
    for offset in [np.array([0, 0, 0]), np.array([0.5, 0.5, 0]), np.array([0.5, 0, 0.5]), np.array([0, 0.5, 0.5])]:
        points = []
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                for k in range(-max_idx, max_idx + 1):
                    p = np.array([i, j, k]) * a + offset * a
                    if len(points) < n:
                        points.append(p)
        
        if len(points) >= n:
            points = np.array(points[:n])
            points -= np.mean(points, axis=0)
            # Add small random perturbation
            points += rng.normal(0, 0.01, (n, 3))
            seeds.append(points)
    
    return seeds

def _generate_hcp_seeds(n, rng):
    """Generate HCP lattice fragments."""
    seeds = []
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    max_idx = int(np.ceil((n / 2) ** (1/3))) + 2
    
    for layer_offset in [0, 0.5]:
        points = []
        for layer in range(-max_idx, max_idx + 1):
            z = layer * c
            for i in range(-max_idx, max_idx + 1):
                for j in range(-max_idx, max_idx + 1):
                    x = (i + j / 2.0) * a
                    y = j * a * np.sqrt(3.0) / 2.0
                    if layer % 2 == 1:
                        x += a / 2.0
                    p = np.array([x, y, z])
                    if len(points) < n:
                        points.append(p)
        
        if len(points) >= n:
            points = np.array(points[:n])
            points -= np.mean(points, axis=0)
            points += rng.normal(0, 0.01, (n, 3))
            seeds.append(points)
    
    return seeds

def _generate_bcc_seeds(n, rng):
    """Generate BCC lattice fragments."""
    seeds = []
    a = 1.0
    max_idx = int(np.ceil((n / 2) ** (1/3))) + 2
    
    for offset in [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]:
        points = []
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                for k in range(-max_idx, max_idx + 1):
                    p = np.array([i, j, k]) * a + offset * a
                    if len(points) < n:
                        points.append(p)
        
        if len(points) >= n:
            points = np.array(points[:n])
            points -= np.mean(points, axis=0)
            points += rng.normal(0, 0.01, (n, 3))
            seeds.append(points)
    
    return seeds

def _generate_icosahedral_seeds(n, rng):
    """Generate icosahedral shell structures."""
    seeds = []
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = [np.array(v) / np.linalg.norm(v) for v in verts]
    
    # Magic numbers for icosahedral shells
    magic = [1, 12, 30, 55, 91]
    
    for r_scale in [1.0, 1.2, 1.5]:
        points = []
        # Center
        if n > 0:
            points.append(np.zeros(3))
        
        # First shell: 12 vertices
        if n > 1:
            for v in verts[:min(12, n - 1)]:
                points.append(v * r_scale)
        
        # Second shell: 30 edge midpoints (approximate)
        if n > 13:
            # Add points on edges
            edges = []
            for i in range(12):
                for j in range(i + 1, 12):
                    dist = np.linalg.norm(verts[i] - verts[j])
                    if dist < 1.5:  # Adjacent vertices
                        edges.append((i, j))
            
            for (i, j) in edges[:min(30, n - 13)]:
                mid = (verts[i] + verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                points.append(mid * r_scale * 1.1)
        
        # Fill remaining
        while len(points) < n:
            angle1 = rng.uniform(0, 2 * np.pi)
            angle2 = rng.uniform(0, np.pi)
            r = r_scale * 1.5
            x = r * np.sin(angle2) * np.cos(angle1)
            y = r * np.sin(angle2) * np.sin(angle1)
            z = r * np.cos(angle2)
            points.append([x, y, z])
        
        points = np.array(points[:n])
        points -= np.mean(points, axis=0)
        points += rng.normal(0, 0.01, (n, 3))
        seeds.append(points)
    
    return seeds

def _generate_random_compact_seeds(n, rng, num_seeds=5):
    """Generate random compact clusters."""
    seeds = []
    radius = n ** (1.0 / 3.0) * 0.8
    
    for _ in range(num_seeds):
        points = []
        min_dist = 0.55
        max_attempts = 50000
        attempts = 0
        
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                attempts += 1
                continue
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
            attempts += 1
        
        while len(points) < n:
            base = points[rng.integers(len(points))] if points else np.zeros(3)
            p = base + rng.uniform(-0.1, 0.1, 3)
            points.append(p)
        
        points = np.array(points)
        points -= np.mean(points, axis=0)
        seeds.append(points)
    
    return seeds

def construct(instance):
    """Find global minimum of Morse cluster via multi-start L-BFGS-B."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse seeds
    seeds = []
    seeds.extend(_generate_fcc_seeds(n, rng))
    seeds.extend(_generate_hcp_seeds(n, rng))
    seeds.extend(_generate_bcc_seeds(n, rng))
    seeds.extend(_generate_icosahedral_seeds(n, rng))
    seeds.extend(_generate_random_compact_seeds(n, rng, num_seeds=3))
    
    # Filter valid seeds
    valid_seeds = []
    for seed in seeds:
        if len(seed) == n:
            # Check min distance
            diff = seed[:, np.newaxis, :] - seed[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                valid_seeds.append(seed)
    
    if not valid_seeds:
        valid_seeds = [_generate_random_compact_seeds(n, rng, num_seeds=1)[0]]
    
    # Run L-BFGS-B on each seed
    best_energy = np.inf
    best_coords = None
    
    for i, seed in enumerate(valid_seeds):
        if _time_remaining() < 5.0:
            break
        
        energy, relaxed = _local_minimize(seed, rho, maxiter=50)
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_coords = relaxed
    
    # If time remains, do perturbation-based additional starts
    if best_coords is not None and _time_remaining() > 10.0:
        for _ in range(10):
            if _time_remaining() < 5.0:
                break
            
            # Perturb best solution
            perturbation_scale = 0.1
            new_seed = best_coords + rng.normal(0, perturbation_scale, (n, 3))
            
            energy, relaxed = _local_minimize(new_seed, rho, maxiter=50)
            
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_coords = relaxed
    
    # Final fallback
    if best_coords is None or not np.isfinite(best_energy):
        coords = _generate_random_compact_seeds(n, rng, num_seeds=1)[0]
        best_energy, best_coords = _local_minimize(coords, rho, maxiter=100)
    
    # Ensure final validity
    diff = best_coords[:, np.newaxis, :] - best_coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        try:
            energy, relaxed = relax(best_coords, 'morse', rho)
            if np.isfinite(energy):
                best_coords = relaxed
        except:
            best_coords = _generate_random_compact_seeds(n, rng, num_seeds=1)[0]
    
    return best_coords
# EVOLVE-BLOCK-END
