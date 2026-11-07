"""Basin Hopping search for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 300.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    """Check that all coordinates are finite and pairwise distances >= min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.all(dists >= min_dist)

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_icosahedron(n):
    """Generate an icosahedral cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    # Normalize to unit radius
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]  # Center atom
    if n > 1:
        pts.extend(v[:min(n-1, 12)])
    
    if n > 13:
        # Add second shell: 30 atoms at edge midpoints + 12 at face centers
        # Edge midpoints of icosahedron
        edges = [
            (0, 1), (0, 2), (0, 4), (0, 5),
            (1, 3), (1, 4), (1, 7), (1, 8),
            (2, 3), (2, 6), (2, 9), (2, 10),
            (3, 7), (3, 8),
            (4, 5), (4, 8), (4, 11),
            (5, 9), (5, 10), (5, 11),
            (6, 7), (6, 9), (6, 10),
            (7, 11),
            (8, 11),
            (9, 10)
        ]
        # Actually, let's just generate points on a sphere for simplicity
        # Second shell radius ~ 1.8
        shell2_radius = 1.8
        for _ in range(min(n - len(pts), 42)):
            # Random point on sphere
            theta = np.random.uniform(0, 2 * np.pi)
            cos_phi = np.random.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = shell2_radius * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            pts.append(p)
    
    return np.array(pts[:n])

def _bh_step(pos, kind, rng, step_size):
    """One basin hopping step: perturb, relax, return (energy, positions)."""
    # Perturb
    noise = rng.normal(0, step_size, pos.shape)
    perturbed = pos + noise
    
    # Relax
    try:
        energy, relaxed = relax(perturbed, 'sutton_chen', kind)
        if not np.isfinite(energy) or not _is_valid(relaxed):
            return np.inf, pos
        return energy, relaxed
    except Exception:
        return np.inf, pos

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Generate initial structures
    generators = [_generate_fcc, _generate_bcc, _generate_icosahedron]
    
    # Try each generator with basin hopping
    for gen in generators:
        if _time_left() < 5.0:
            break
        
        base = gen(n)
        if not _is_valid(base):
            continue
        
        # Initial relaxation
        try:
            energy, pos = relax(base, 'sutton_chen', kind)
            if not np.isfinite(energy) or not _is_valid(pos):
                continue
        except Exception:
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
        
        # Basin hopping from this starting point
        step_size = 0.3
        for step in range(50):
            if _time_left() < 5.0:
                break
            
            new_energy, new_pos = _bh_step(pos, kind, rng, step_size)
            
            if new_energy < energy:
                pos = new_pos
                energy = new_energy
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            else:
                # Metropolis acceptance with T=0.1
                if rng.uniform() < np.exp(-(new_energy - energy) / 0.1):
                    pos = new_pos
                    energy = new_energy
                else:
                    # Try again with slightly different perturbation
                    continue
    
    # If no valid position found, return a default valid one
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            # Try BCC
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            # Last resort: random with minimum distance enforcement
            best_pos = rng.uniform(-2, 2, (n, 3))
            # Ensure minimum distance
            for _ in range(100):
                if _is_valid(best_pos):
                    break
                best_pos = rng.uniform(-2, 2, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
