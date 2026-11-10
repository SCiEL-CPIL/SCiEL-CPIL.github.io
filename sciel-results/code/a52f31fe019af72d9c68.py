"""Morse cluster global minima: basin hopping with crystalline templates and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _generate_fcc(n):
    """Generate n atoms from an FCC lattice, centered."""
    a = 1.0  # lattice constant = equilibrium distance
    # FCC primitive vectors
    v1 = np.array([0, a/2, a/2])
    v2 = np.array([a/2, 0, a/2])
    v3 = np.array([a/2, a/2, 0])
    
    # Generate FCC lattice points in a cube
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    points = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                # FCC basis
                for bx, by, bz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i, j, k]) * a + np.array([bx, by, bz]) * a
                    points.append(p)
    
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    
    # Sort by distance from center and take first n
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    points = points[idx]
    
    return points


def _generate_bcc(n):
    """Generate n atoms from a BCC lattice, centered."""
    a = 1.0
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    points = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                p = np.array([i, j, k]) * a
                points.append(p)
                # BCC center
                p2 = p + np.array([0.5, 0.5, 0.5]) * a
                points.append(p2)
    
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_hcp(n):
    """Generate n atoms from an HCP-like lattice, centered."""
    a = 1.0
    c = np.sqrt(8.0/3.0) * a  # ideal HCP c/a
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    points = []
    for layer in range(-side, side+1):
        z = layer * c
        offset = 0.5 * a if layer % 2 == 0 else 0.0
        for i in range(-side, side+1):
            for j in range(-side, side+1):
                x = (i + offset) * a
                y = j * a * np.sqrt(3.0) / 2.0
                points.append(np.array([x, y, z]))
    
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _random_cluster(n, rng):
    """Generate a random cluster with minimum spacing."""
    spacing = 0.8
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_tries = n * 100
    tries = 0
    while len(points) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= spacing for q in points):
                points.append(p)
        tries += 1
    # If we couldn't place all, fill with random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _basin_hopping(pos, rho, n, rng, max_iter=50, step_size=0.5, deadline=None):
    """Basin hopping optimization."""
    best_pos = pos.copy()
    best_energy, _ = relax(pos, 'morse', rho)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        noise = rng.normal(0, step_size, pos.shape)
        new_pos = pos + noise
        
        # Relax
        new_energy, new_pos = relax(new_pos, 'morse', rho)
        
        # Metropolis accept
        delta_e = new_energy - best_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 1.0):
            pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster configurations using basin hopping from diverse starts."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    deadline = time.time() + 280  # 280s safety margin
    
    rng = np.random.default_rng(hash((n, rho)) % (2**32))
    
    # Generate diverse starting configurations
    starts = []
    
    # Crystalline templates
    try:
        starts.append(_generate_fcc(n))
    except:
        pass
    try:
        starts.append(_generate_bcc(n))
    except:
        pass
    try:
        starts.append(_generate_hcp(n))
    except:
        pass
    
    # Random starts
    for _ in range(3):
        starts.append(_random_cluster(n, rng))
    
    # Basin hop from each start, keeping the best
    best_energy = np.inf
    best_pos = starts[0]
    
    # Distribute time budget across starts
    n_starts = len(starts)
    time_per_start = (280.0 / n_starts) * 0.9  # leave 10% margin
    
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Adaptive max iterations based on time remaining
        elapsed = time.time() - (deadline - 280)
        remaining = max(0, 280 - elapsed)
        max_iter = max(10, int(remaining / (time_per_start / 50.0)))
        
        energy, pos = _basin_hopping(start, rho, n, rng, max_iter=max_iter, 
                                       step_size=0.4, deadline=deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Final check
    if time.time() > deadline - 5:
        # Return immediately
        pass
    
    return best_pos
# EVOLVE-BLOCK-END
