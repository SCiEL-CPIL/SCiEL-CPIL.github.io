"""Morse cluster global minima via lattice initialization and multi-restart relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import morse_energy_gradient

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 290.0
    return _T0


def _time_left():
    global _HARD_DEADLINE
    return max(0.0, _HARD_DEADLINE - time.time())


def _morse_energy(positions, rho):
    """Compute Morse energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _enforce_min_dist(P, min_d=0.5):
    """Ensure all pair distances >= min_d by small local pushes."""
    P = np.array(P, dtype=float)
    n = P.shape[0]
    for _ in range(30):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                d = P[i] - P[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-12:
                        d = np.array([1.0, 0, 0])
                    else:
                        d = d / dist
                    push = (min_d - dist) / 2.0 + 0.005
                    P[i] += push * d
                    P[j] -= push * d
                    ok = False
        if ok:
            break
    return P


def _fcc_points(n, spacing=1.0):
    """Generate n points from FCC lattice centered at origin."""
    points = []
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * spacing
    
    r_max = spacing * (n ** (1.0 / 3.0) + 2)
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


def _hcp_points(n, spacing=1.0):
    """Generate n points from an HCP lattice, centered."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0)/2.0 * j
                z = c * (k + (1.0/3.0) * ((j % 2) + 1))
                pts.append(np.array([x, y, z]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _bcc_points(n, spacing=1.0):
    """Generate n points from BCC lattice centered at origin."""
    r_max = spacing * (n ** (1.0 / 3.0) + 2)
    
    points = []
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


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice, centered."""
    pts = []
    a = spacing
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                pts.append(np.array([i, j, k]) * a)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


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
    _init_timer()
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(size * 1000 + int(rho))
    
    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    
    # Lattice initializations
    try:
        inits.append(_fcc_points(size, spacing=1.0))
    except Exception:
        pass
    
    try:
        inits.append(_hcp_points(size, spacing=1.0))
    except Exception:
        pass
    
    try:
        inits.append(_bcc_points(size, spacing=1.0))
    except Exception:
        pass
    
    try:
        inits.append(_sc_points(size, spacing=1.0))
    except Exception:
        pass
    
    # Random compact clusters
    for seed_offset in [0, 7, 13, 23, 42]:
        try:
            rng_sub = np.random.default_rng(size * 1000 + int(rho) + seed_offset)
            inits.append(_random_compact(size, spacing=1.0, rng=rng_sub))
        except Exception:
            pass
    
    # Perturbed FCC variants
    if len(inits) > 0:
        for scale in [0.08, 0.15, 0.2]:
            try:
                perturbed = _perturb(inits[0].copy(), rng, scale=scale)
                inits.append(perturbed)
            except Exception:
                pass
    
    # Try each initialization
    for init_idx, init_pos in enumerate(inits):
        if _time_left() < 5.0:
            break
        
        try:
            init_pos = _enforce_min_dist(init_pos, 0.5)
            energy, positions = relax(init_pos, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Phase 2: Perturb-best-and-relax cycles with decreasing scales
    if best_positions is not None and _time_left() > 5.0:
        scales = [0.25, 0.18, 0.12, 0.08, 0.05, 0.03]
        for scale in scales:
            if _time_left() < 4.0:
                break
            n_tries = 3 if scale > 0.1 else 4
            for _ in range(n_tries):
                if _time_left() < 4.0:
                    break
                try:
                    perturbed = _perturb(best_positions.copy(), rng, scale=scale)
                    perturbed = _enforce_min_dist(perturbed, 0.5)
                    energy, positions = relax(perturbed, potential, rho)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                except Exception:
                    continue
    
    # Phase 3: Larger perturbations if time remains
    if best_positions is not None and _time_left() > 8.0:
        for scale in [0.4, 0.5]:
            if _time_left() < 4.0:
                break
            for _ in range(2):
                if _time_left() < 4.0:
                    break
                try:
                    perturbed = _perturb(best_positions.copy(), rng, scale=scale)
                    perturbed = _enforce_min_dist(perturbed, 0.5)
                    energy, positions = relax(perturbed, potential, rho)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                except Exception:
                    continue
    
    # Fallback: if no valid result, return a simple compact cluster
    if best_positions is None:
        best_positions = _random_compact(size, spacing=1.0, rng=rng)
        best_positions = _enforce_min_dist(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
