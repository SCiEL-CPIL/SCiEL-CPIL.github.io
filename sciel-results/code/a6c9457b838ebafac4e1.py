"""Morse cluster global minima via lattice initialization and multi-restart relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute Morse energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Mask upper triangle
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _fcc_points(n, spacing=1.0):
    """Generate n points from FCC lattice centered at origin."""
    # FCC lattice points in a cube
    # Generate candidate points
    r_max = spacing * (n ** (1.0 / 3.0) + 2)
    # Use a grid approach for FCC
    # FCC has 4 atoms per cubic unit cell
    # We'll generate points from FCC and select the closest n to origin
    
    points = []
    # Generate FCC lattice points
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * spacing
    
    # Generate grid points
    grid_size = int(r_max / spacing) + 3
    for ix in range(-grid_size, grid_size + 1):
        for iy in range(-grid_size, grid_size + 1):
            for iz in range(-grid_size, grid_size + 1):
                for b in basis:
                    p = np.array([ix, iy, iz]) * spacing + b
                    if np.linalg.norm(p) <= r_max:
                        points.append(p)
    
    points = np.array(points)
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    # Select n points
    if len(points) >= n:
        points = points[:n]
    else:
        # Pad with random points if needed
        while len(points) < n:
            p = np.random.uniform(-r_max, r_max, 3)
            points = np.vstack([points, p])
    
    # Center at origin
    points -= np.mean(points, axis=0)
    return points


def _bcc_points(n, spacing=1.0):
    """Generate n points from BCC lattice centered at origin."""
    r_max = spacing * (n ** (1.0 / 3.0) + 2)
    
    points = []
    # BCC basis: (0,0,0) and (0.5,0.5,0.5)
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]) * spacing
    
    grid_size = int(r_max / spacing) + 3
    for ix in range(-grid_size, grid_size + 1):
        for iy in range(-grid_size, grid_size + 1):
            for iz in range(-grid_size, grid_size + 1):
                for b in basis:
                    p = np.array([ix, iy, iz]) * spacing + b
                    if np.linalg.norm(p) <= r_max:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    if len(points) >= n:
        points = points[:n]
    else:
        while len(points) < n:
            p = np.random.uniform(-r_max, r_max, 3)
            points = np.vstack([points, p])
    
    points -= np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=1.0, rng=None):
    """Generate random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = 10000
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    
    # If we couldn't place all points, fill randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _perturb(positions, rng, scale=0.3):
    """Add random perturbation to positions."""
    noise = rng.normal(0, scale, size=positions.shape)
    return positions + noise


def construct(instance):
    """Find low-energy Morse cluster configurations using lattice init and multi-restart relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    start_time = time.time()
    deadline = start_time + 45.0  # ~45s per instance, 6 instances = 270s total
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(size * 1000 + int(rho))
    
    inits = []
    
    # FCC lattice
    try:
        inits.append(_fcc_points(size, spacing=1.0))
    except Exception:
        pass
    
    # BCC lattice
    try:
        inits.append(_bcc_points(size, spacing=1.0))
    except Exception:
        pass
    
    # Random compact clusters
    for _ in range(4):
        try:
            inits.append(_random_compact(size, spacing=1.0, rng=rng))
        except Exception:
            pass
    
    # Perturbed FCC variants
    if len(inits) > 0:
        for _ in range(2):
            try:
                perturbed = _perturb(inits[0].copy(), rng, scale=0.2)
                inits.append(perturbed)
            except Exception:
                pass
    
    # Try each initialization
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline:
            break
        
        try:
            energy, positions = relax(init_pos, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # If we have time, do additional perturbation-restart cycles
    if best_positions is not None and time.time() < deadline - 2.0:
        for _ in range(3):
            if time.time() > deadline:
                break
            try:
                perturbed = _perturb(best_positions.copy(), rng, scale=0.15)
                energy, positions = relax(perturbed, potential, rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    # Fallback: if no valid result, return a simple compact cluster
    if best_positions is None:
        best_positions = _random_compact(size, spacing=1.0, rng=rng)
    
    return best_positions
# EVOLVE-BLOCK-END
