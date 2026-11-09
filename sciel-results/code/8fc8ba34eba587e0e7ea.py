"""Sutton-Chen cluster global-minima search: diverse structural prototypes + multi-restart relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = time.time() + 280.0


def _remaining():
    return _DEADLINE - time.time()


def _valid(pos):
    """Check if positions are valid: finite, no two atoms closer than 0.35."""
    if pos is None or not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    if n < 2:
        return True
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min() > 0.35


def _fcc_cluster(n):
    """Generate an FCC-like cluster of n atoms."""
    # Generate FCC lattice points
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    all_pts = []
    for p in pts:
        for b in basis:
            all_pts.append(p + b)
    all_pts = np.array(all_pts)
    
    # Sort by distance from origin
    dists = np.linalg.norm(all_pts, axis=1)
    order = np.argsort(dists)
    all_pts = all_pts[order]
    
    # Take the n closest
    if len(all_pts) >= n:
        cluster = all_pts[:n]
    else:
        # Pad with random points if not enough
        rng = np.random.default_rng(42)
        extra = rng.uniform(-1, 1, (n - len(all_pts), 3))
        cluster = np.vstack([all_pts, extra])
    
    # Center
    cluster = cluster - cluster.mean(axis=0)
    return cluster


def _bcc_cluster(n):
    """Generate a BCC-like cluster of n atoms."""
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float)
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    if len(pts) >= n:
        cluster = pts[:n]
    else:
        rng = np.random.default_rng(43)
        extra = rng.uniform(-1, 1, (n - len(pts), 3))
        cluster = np.vstack([pts, extra])
    
    cluster = cluster - cluster.mean(axis=0)
    return cluster


def _hcp_cluster(n):
    """Generate an HCP-like cluster of n atoms."""
    pts = []
    a = 1.0
    c = 1.633 * a  # ideal HCP c/a ratio
    # HCP: two layers per unit cell, A-B stacking
    for layer in range(-4, 5):
        z = layer * c / 2
        for i in range(-4, 5):
            for j in range(-4, 5):
                if layer % 2 == 0:
                    x = i * a + 0.5 * (j % 2) * a
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + 0.5 * a
                    y = j * a * np.sqrt(3) / 2
                pts.append([x, y, z])
    pts = np.array(pts, dtype=float)
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    if len(pts) >= n:
        cluster = pts[:n]
    else:
        rng = np.random.default_rng(44)
        extra = rng.uniform(-1, 1, (n - len(pts), 3))
        cluster = np.vstack([pts, extra])
    
    cluster = cluster - cluster.mean(axis=0)
    return cluster


def _random_cluster(n, seed, spacing=0.72):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all atoms, add random ones
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _perturb(cluster, seed, scale=0.1):
    """Add small random perturbation to a cluster."""
    rng = np.random.default_rng(seed)
    return cluster + rng.normal(0, scale, cluster.shape)


def construct(instance):
    """Best of diverse structural prototypes, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting points
    starts = []
    
    # FCC prototype
    try:
        fcc = _fcc_cluster(size)
        if _valid(fcc):
            starts.append(fcc)
    except:
        pass
    
    # BCC prototype
    try:
        bcc = _bcc_cluster(size)
        if _valid(bcc):
            starts.append(bcc)
    except:
        pass
    
    # HCP prototype
    try:
        hcp = _hcp_cluster(size)
        if _valid(hcp):
            starts.append(hcp)
    except:
        pass
    
    # Random compact clusters with different seeds
    for seed in [size, size + 1, size + 2, size + 3]:
        try:
            rand = _random_cluster(size, seed)
            if _valid(rand):
                starts.append(rand)
        except:
            pass
    
    # Perturbed FCC
    try:
        fcc = _fcc_cluster(size)
        perturbed = _perturb(fcc, size + 10, scale=0.05)
        if _valid(perturbed):
            starts.append(perturbed)
    except:
        pass
    
    # Perturbed BCC
    try:
        bcc = _bcc_cluster(size)
        perturbed = _perturb(bcc, size + 20, scale=0.05)
        if _valid(perturbed):
            starts.append(perturbed)
    except:
        pass
    
    # Relax each starting point
    for i, start in enumerate(starts):
        if _remaining() < 10.0:
            break
        try:
            energy, positions = relax(start, potential, parameter)
            if _valid(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    # If we still have time, try more random starts
    extra_seeds = [size + 100, size + 200, size + 300, size + 400]
    for seed in extra_seeds:
        if _remaining() < 10.0:
            break
        try:
            rand = _random_cluster(size, seed, spacing=0.65)
            if _valid(rand):
                energy, positions = relax(rand, potential, parameter)
                if _valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except:
            pass
    
    # If we still have time, try more perturbations of the best found
    if best_positions is not None and _remaining() > 15.0:
        for seed in [size + 500, size + 600]:
            if _remaining() < 10.0:
                break
            try:
                perturbed = _perturb(best_positions, seed, scale=0.08)
                if _valid(perturbed):
                    energy, positions = relax(perturbed, potential, parameter)
                    if _valid(positions) and energy < best_energy:
                        best_energy = energy
                        best_positions = positions
            except:
                pass
    
    # Fallback: if no valid result, return a simple linear arrangement
    if best_positions is None:
        best_positions = np.zeros((size, 3))
        for i in range(size):
            best_positions[i] = [i * 0.4, 0, 0]
    
    return best_positions
# EVOLVE-BLOCK-END
