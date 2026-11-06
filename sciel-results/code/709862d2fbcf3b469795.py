"""Morse cluster global minima via basin hopping with diverse seeds."""
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
        _HARD_DEADLINE = _T0 + 285.0
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
    """Ensure all pair distances >= min_d by iterative local pushes."""
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    if n < 2:
        return P
    for _ in range(50):
        moved = False
        # Vectorized check for violations
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        # Find pairs below min_d
        mask = dist < min_d
        if not np.any(mask):
            break
        # Process violations
        iu = np.triu_indices(n, k=1)
        for i, j in zip(iu[0][mask[iu]], iu[1][mask[iu]]):
            d = P[i] - P[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij < 1e-12:
                d = np.array([1.0, 0, 0])
            else:
                d = d / dist_ij
            push = (min_d - dist_ij) / 2.0 + 0.002
            P[i] += push * d
            P[j] -= push * d
            moved = True
        if not moved:
            break
    return P


def _fcc_points(n, spacing=1.0):
    """Generate n points from FCC lattice centered at origin."""
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * spacing
    
    r_max = spacing * (n ** (1.0 / 3.0) + 2)
    grid_size = int(r_max / spacing) + 3
    points = []
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
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]) * spacing
    
    grid_size = int(r_max / spacing) + 3
    points = []
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
    max_attempts = 20000
    
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


def _icosahedral_shells(n, spacing=1.0):
    """Generate points on icosahedral shells."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, -phi, 0], [1, -phi, 0], [-1, phi, 0],
        [phi, 0, 1], [-phi, 0, -1], [phi, 0, -1], [-phi, 0, 1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) * spacing
    
    # Generate points on shells
    points = [np.zeros(3)]
    shell_r = spacing
    shell_idx = 1
    
    while len(points) < n:
        # Add points on current shell
        n_on_shell = min(12 * shell_idx * shell_idx, n - len(points))
        for _ in range(n_on_shell):
            # Random point on sphere
            v = np.random.randn(3)
            v = v / np.linalg.norm(v) * shell_r
            points.append(v)
        
        if len(points) >= n:
            break
        shell_r += spacing * 0.8
        shell_idx += 1
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def construct(instance):
    """Find low-energy Morse cluster configurations using basin hopping."""
    _init_timer()
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(size * 1000 + int(rho))
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial configurations
    inits = []
    
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
    
    try:
        inits.append(_icosahedral_shells(size, spacing=1.0))
    except Exception:
        pass
    
    # Random compact clusters with different seeds
    for seed_offset in [0, 7, 13, 23, 42, 99, 123]:
        try:
            rng_sub = np.random.default_rng(size * 1000 + int(rho) + seed_offset)
            inits.append(_random_compact(size, spacing=1.0, rng=rng_sub))
        except Exception:
            pass
    
    # Phase 1: Relax all initial configurations
    for init_pos in inits:
        if _time_left() < 3.0:
            break
        try:
            init_pos = _enforce_min_dist(init_pos, 0.5)
            energy, positions = relax(init_pos, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback
        best_positions = _random_compact(size, spacing=1.0, rng=rng)
        best_positions = _enforce_min_dist(best_positions, 0.5)
        try:
            best_energy, best_positions = relax(best_positions, potential, rho)
        except Exception:
            best_energy = _morse_energy(best_positions, rho)
    
    # Phase 2: Basin hopping with adaptive kick scales
    if _time_left() > 5.0:
        # Start with moderate kicks, adapt based on acceptance
        kick_scale = 0.25
        min_kick = 0.02
        max_kick = 0.6
        n_stagnant = 0
        max_stagnant = 8
        
        while _time_left() > 3.0:
            # Generate perturbed configuration
            noise = rng.normal(0, kick_scale, size=best_positions.shape)
            perturbed = best_positions + noise
            perturbed = _enforce_min_dist(perturbed, 0.5)
            
            try:
                energy, positions = relax(perturbed, potential, rho)
            except Exception:
                kick_scale *= 0.8
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                n_stagnant = 0
                # Accept: slightly increase kick to explore
                kick_scale = min(kick_scale * 1.05, max_kick)
            else:
                n_stagnant += 1
                # Reject: decrease kick to fine-tune
                kick_scale = max(kick_scale * 0.9, min_kick)
            
            if n_stagnant >= max_stagnant:
                # Try a larger kick to escape local minimum
                kick_scale = min(kick_scale * 2.0, max_kick)
                n_stagnant = 0
    
    # Phase 3: Fine-tuning with very small kicks if time remains
    if best_positions is not None and _time_left() > 5.0:
        for _ in range(20):
            if _time_left() < 2.0:
                break
            try:
                noise = rng.normal(0, 0.03, size=best_positions.shape)
                perturbed = best_positions + noise
                perturbed = _enforce_min_dist(perturbed, 0.5)
                energy, positions = relax(perturbed, potential, rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    return best_positions
# EVOLVE-BLOCK-END
