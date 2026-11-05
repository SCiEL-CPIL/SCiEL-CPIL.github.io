"""
Strategy: Crystalline template generation + multi-start L-BFGS-B refinement.
Uses FCC/BCC/HCP lattice truncations as high-quality starting points,
then refines each with L-BFGS-B for precise local minima.
Deadline guard ensures all 6 instances finish within 330s.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # safety margin before 330s kill

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(positions, rho):
    """Compute Morse potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Vectorized pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, 1)
    r = dist[iu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)

def _morse_gradient(positions, rho):
    """Compute gradient of Morse potential energy."""
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    iu = np.triu_indices(n, 1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff ** 2, axis=1))
    # Avoid division by zero
    mask = r > 1e-10
    r_safe = np.where(mask, r, 1.0)
    x = rho * (1.0 - r_safe)
    e = np.exp(x)
    # dE/dr = -2*rho*e*(e-1)
    dEdr = -2.0 * rho * e * (e - 1.0)
    # grad_i = -dEdr * (ri - rj)/r, grad_j = dEdr * (ri - rj)/r
    factor = dEdr / r_safe
    factor = np.where(mask, factor, 0.0)
    np.add.at(grad, iu[0], -factor[:, np.newaxis] * diff)
    np.add.at(grad, iu[1], factor[:, np.newaxis] * diff)
    return grad

def _generate_fcc(n, spacing=1.0):
    """Generate n atoms in FCC-like arrangement."""
    atoms = []
    # FCC basis vectors
    a = spacing * 2.0  # lattice constant for equilibrium distance 1
    # Generate FCC lattice points
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC: simple cubic + face-centered
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    atoms.append([x, y, z])
    atoms = np.array(atoms)
    if len(atoms) >= n:
        # Take the n closest to origin
        dists = np.sum(atoms ** 2, axis=1)
        idx = np.argsort(dists)[:n]
        return atoms[idx]
    else:
        # Pad with random points if not enough
        while len(atoms) < n:
            atoms = np.vstack([atoms, np.random.randn(3) * 0.5])
        return atoms[:n]

def _generate_bcc(n, spacing=1.0):
    """Generate n atoms in BCC-like arrangement."""
    atoms = []
    a = spacing * 2.0
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    atoms.append([x, y, z])
    atoms = np.array(atoms)
    if len(atoms) >= n:
        dists = np.sum(atoms ** 2, axis=1)
        idx = np.argsort(dists)[:n]
        return atoms[idx]
    else:
        while len(atoms) < n:
            atoms = np.vstack([atoms, np.random.randn(3) * 0.5])
        return atoms[:n]

def _generate_hcp(n, spacing=1.0):
    """Generate n atoms in HCP-like arrangement."""
    atoms = []
    a = spacing * 2.0
    c = a * math.sqrt(2.0 / 3.0)
    for k in range(-4, 5):
        for i in range(-4, 5):
            for j in range(-4, 5):
                # HCP: AB stacking
                if k % 2 == 0:
                    x = (i + 0.5 * (j % 2)) * a
                    y = (j * math.sqrt(3) / 2.0) * a
                else:
                    x = (i + 0.5 * (j % 2) + 1.0/3.0) * a
                    y = (j * math.sqrt(3) / 2.0 + 1.0/3.0) * a
                z = k * c
                atoms.append([x, y, z])
    atoms = np.array(atoms)
    if len(atoms) >= n:
        dists = np.sum(atoms ** 2, axis=1)
        idx = np.argsort(dists)[:n]
        return atoms[idx]
    else:
        while len(atoms) < n:
            atoms = np.vstack([atoms, np.random.randn(3) * 0.5])
        return atoms[:n]

def _random_cluster(n, spacing=1.0, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = n * 50
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        attempt += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
    # If not enough points, add random ones
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _jitter_crystal(positions, jitter=0.1):
    """Add small random jitter to a crystal structure."""
    return positions + np.random.randn(*positions.shape) * jitter

def _refine(positions, rho, max_iter=500, time_limit=30.0):
    """Refine positions using L-BFGS-B."""
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def jac(x):
        pos = x.reshape(-1, 3)
        g = _morse_gradient(pos, rho)
        return g.flatten()
    
    x0 = positions.flatten()
    start = time.time()
    
    def callback(xk):
        return time.time() - start > time_limit
    
    try:
        result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'callback': callback})
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return positions, _morse_energy(positions, rho)

def construct(instance):
    """Find low-energy arrangement of n atoms with Morse potential."""
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    potential = instance["potential"]
    
    remaining = _get_remaining_time()
    if remaining < 5.0:
        # Emergency: return a quick random cluster
        return _random_cluster(n, 1.0, seed=n)
    
    # Time budget per refinement
    time_per_refine = max(2.0, min(40.0, remaining / 12.0))
    
    # Generate diverse starting points
    starts = []
    
    # Crystalline templates
    starts.append(_generate_fcc(n, 1.0))
    starts.append(_generate_bcc(n, 1.0))
    starts.append(_generate_hcp(n, 1.0))
    
    # Jittered versions
    starts.append(_jitter_crystal(_generate_fcc(n, 1.0), 0.05))
    starts.append(_jitter_crystal(_generate_bcc(n, 1.0), 0.05))
    starts.append(_jitter_crystal(_generate_hcp(n, 1.0), 0.05))
    
    # Random clusters
    for seed in [n, n+1, n+2, n+3]:
        starts.append(_random_cluster(n, 1.0, seed=seed))
    
    # Additional jittered random
    for seed in [n+10, n+20]:
        base = _random_cluster(n, 1.0, seed=seed)
        starts.append(_jitter_crystal(base, 0.1))
    
    best_energy = np.inf
    best_positions = None
    
    for i, start_pos in enumerate(starts):
        remaining = _get_remaining_time()
        if remaining < 5.0:
            break
        
        # Ensure minimum separation
        # Quick check and fix if needed
        pos = start_pos.copy()
        
        # Refine
        refined_pos, energy = _refine(pos, rho, max_iter=300, time_limit=time_per_refine)
        
        # Verify energy
        if energy < best_energy:
            best_energy = energy
            best_positions = refined_pos
        
        # Also try the relax function on the best so far occasionally
        if i % 3 == 0:
            remaining = _get_remaining_time()
            if remaining > 10.0 and best_positions is not None:
                try:
                    e2, pos2 = relax(best_positions.copy(), potential, rho)
                    if e2 < best_energy:
                        best_energy = e2
                        best_positions = pos2
                except Exception:
                    pass
    
    if best_positions is None:
        best_positions = starts[0]
    
    # Final verification and possible additional refinement
    remaining = _get_remaining_time()
    if remaining > 10.0:
        final_pos, final_e = _refine(best_positions, rho, max_iter=200, time_limit=min(remaining - 5, 20.0))
        if final_e < best_energy:
            best_positions = final_pos
            best_energy = final_e
    
    return best_positions
# EVOLVE-BLOCK-END
