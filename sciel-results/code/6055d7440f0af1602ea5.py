"""Morse cluster global minima: basin hopping with structured initializations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 280.0  # safety margin before 300s kill
_DEADLINE = _START + _BUDGET


def _remaining():
    return max(0.0, _DEADLINE - time.time())


def _morse_energy(rho, coords):
    """Exact Morse energy; returns large penalty if any pair < 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return 1e10
    e = np.exp(rho * (1.0 - d))
    return np.sum(e * (e - 2.0)) / 2.0


def _perturb(coords, rng, scale=0.15):
    """Random displacement of a random subset of atoms."""
    n = len(coords)
    k = max(1, int(n * 0.1))
    idx = rng.choice(n, size=k, replace=False)
    noise = rng.normal(0.0, scale, size=(k, 3))
    new = coords.copy()
    new[idx] += noise
    return new


def _basin_hopping(rho, x0, rng, max_iter=60, temp=0.05):
    """Simple basin hopping: perturb, relax, Metropolis accept."""
    best_x = x0.copy()
    best_e = _morse_energy(rho, best_x)
    cur_e = best_e
    cur_x = best_x.copy()
    for _ in range(max_iter):
        if _remaining() < 2.0:
            break
        x_new = _perturb(cur_x, rng)
        # Local relax
        e_rel, x_rel = relax(x_new, 'morse', rho)
        # Validate
        if e_rel > 1e8:
            continue
        dE = e_rel - cur_e
        if dE < 0 or rng.random() < np.exp(-dE / temp):
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
    return best_x, best_e


def _sphere_shell(n, radius, rng):
    """Place n points on a sphere of given radius (Fibonacci-like)."""
    if n == 0:
        return np.zeros((0, 3))
    golden = (1 + np.sqrt(5)) / 2
    pts = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(max(0, 1 - z * z))
        theta = 2 * np.pi * i / golden
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    pts = np.array(pts) * radius
    # Add small jitter to break perfect symmetry
    pts += rng.normal(0, 0.02, pts.shape)
    return pts


def _icosahedral_seed(n, rng):
    """Icosahedral shell + inner atoms for larger n."""
    # 12 icosahedral vertices
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])  # normalize to unit sphere
    ico *= 1.0  # radius 1
    
    if n <= 12:
        pts = ico[:n].copy()
        pts += rng.normal(0, 0.03, pts.shape)
        return pts
    
    remaining = n - 12
    # Place remaining atoms in inner shell
    inner_r = 0.5
    inner = _sphere_shell(remaining, inner_r, rng)
    pts = np.vstack([ico, inner])
    return pts


def _fcc_seed(n, rng):
    """FCC-like packing in a cube."""
    # Generate FCC lattice points
    a = 1.0  # lattice parameter
    points = []
    # Bulk FCC
    for i in range(0, 5):
        for j in range(0, 5):
            for k in range(0, 5):
                # Corners
                points.append([i * a, j * a, k * a])
                # Face centers
                if i < 4:
                    points.append([(i + 0.5) * a, j * a, k * a])
                if j < 4:
                    points.append([i * a, (j + 0.5) * a, k * a])
                if k < 4:
                    points.append([i * a, j * a, (k + 0.5) * a])
    points = np.array(points)
    # Center
    points = points - points.mean(axis=0)
    # Sort by distance from center and take n closest
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    pts = points[order[:n]].copy()
    pts += rng.normal(0, 0.03, pts.shape)
    return pts


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse initializations
    seeds = []
    
    # 1. Icosahedral-based
    seeds.append(_icosahedral_seed(n, rng))
    
    # 2. FCC-based
    seeds.append(_fcc_seed(n, rng))
    
    # 3. Spherical shell (single)
    r1 = n ** (1/3) * 0.5
    seeds.append(_sphere_shell(n, r1, rng))
    
    # 4. Two shells
    if n > 12:
        outer_n = max(12, int(n * 0.6))
        inner_n = n - outer_n
        outer = _sphere_shell(outer_n, 1.2, rng)
        inner = _sphere_shell(inner_n, 0.5, rng)
        seeds.append(np.vstack([outer, inner]))
    
    # 5. Random compact
    radius = n ** (1/3) * 0.8
    pts = []
    for _ in range(n):
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            pts.append(p)
    seeds.append(np.array(pts))
    
    # Basin hopping from each seed
    best_x = None
    best_e = np.inf
    
    for seed in seeds:
        if _remaining() < 5.0:
            break
        # First relax the seed
        e0, x0 = relax(seed, 'morse', rho)
        if e0 > 1e8:
            continue
        
        # Basin hopping
        x_bh, e_bh = _basin_hopping(rho, x0, rng, max_iter=40, temp=0.03)
        
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh
    
    # Final relaxation
    if best_x is not None:
        _, best_x = relax(best_x, 'morse', rho)
        # Validate
        if _morse_energy(rho, best_x) < 1e8:
            return best_x
    
    # Fallback: return the best seed relaxed
    for seed in seeds:
        e, x = relax(seed, 'morse', rho)
        if e < 1e8:
            return x
    
    # Last resort
    return seeds[0]
# EVOLVE-BLOCK-END
