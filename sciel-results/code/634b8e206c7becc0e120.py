"""Morse cluster global minima: diverse initializations + parallel relaxation + polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE = 330.0  # wall-clock limit; use safety margin
_MARGIN = 15.0     # seconds to leave before kill

def _morse_energy(positions, rho):
    """Compute total Morse energy: sum exp(rho(1-r))(exp(rho(1-r))-2)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist = np.maximum(dist, 1e-10)
    x = rho * (1.0 - dist)
    ex = np.exp(x)
    pair_e = ex * (ex - 2.0)
    return float(np.sum(pair_e) / 2.0)

def _is_valid(positions, min_dist=0.5):
    """Check all pairwise distances >= min_dist."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist) >= min_dist

def _generate_fcc(n, rng):
    """Generate FCC-like cluster: fill FCC lattice sites nearest to origin."""
    # FCC lattice vectors
    a = 1.0  # equilibrium spacing
    sites = []
    # Generate FCC sites in a box
    max_idx = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for bx, by, bz in [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)]:
                    site = np.array([i*0.5, j*0.5, k*0.5]) + np.array([bx, by, bz])
                    sites.append(site)
    sites = np.array(sites)
    # Sort by distance from origin, take first n
    dists = np.linalg.norm(sites, axis=1)
    order = np.argsort(dists)
    selected = sites[order[:n]]
    # Center
    selected = selected - np.mean(selected, axis=0)
    # Add small random perturbation to break symmetry
    selected += rng.normal(0, 0.01, selected.shape)
    return selected

def _generate_hcp(n, rng):
    """Generate HCP-like cluster."""
    a = 1.0
    c_over_a = np.sqrt(8.0/3.0)
    sites = []
    max_idx = int(np.ceil(n ** (1/3))) + 3
    for layer in range(-max_idx, max_idx+1):
        z = layer * c_over_a * a / 2.0
        offset = (0.5, 0.0) if layer % 2 == 0 else (0.0, 0.0)
        for i in range(-max_idx, max_idx+1):
            for j in range(-max_idx, max_idx+1):
                x = (i + offset[0]) * a
                y = (j + offset[1]) * a
                sites.append(np.array([x, y, z]))
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    order = np.argsort(dists)
    selected = sites[order[:n]]
    selected = selected - np.mean(selected, axis=0)
    selected += rng.normal(0, 0.01, selected.shape)
    return selected

def _generate_icosahedral(n, rng):
    """Generate icosahedral shell cluster (good for small n)."""
    # Icosahedron vertices (radius 1)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi]
    ])
    # Normalize
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    # Remove duplicates
    ico_verts = np.unique(np.round(ico_verts, 10), axis=0)
    
    sites = [np.zeros(3)]  # center atom
    # Shell 1: icosahedron vertices at radius ~1
    shell1 = ico_verts * 1.0
    sites.extend(shell1.tolist())
    # Shell 2: more vertices at radius ~2
    if n > 13:
        # Use icosahedron vertices at radius 2 with small perturbation
        shell2 = ico_verts * 2.0 + rng.normal(0, 0.05, (len(ico_verts), 3))
        sites.extend(shell2.tolist())
    # Shell 3 if needed
    if n > 25:
        shell3 = ico_verts * 3.0 + rng.normal(0, 0.05, (len(ico_verts), 3))
        sites.extend(shell3.tolist())
    if n > 37:
        shell4 = ico_verts * 4.0 + rng.normal(0, 0.05, (len(ico_verts), 3))
        sites.extend(shell4.tolist())
    
    sites = np.array(sites[:n])
    if len(sites) < n:
        # Fill remaining with random points
        while len(sites) < n:
            p = rng.normal(0, 1, 3) * 2
            sites = np.vstack([sites, p])
    sites = sites - np.mean(sites, axis=0)
    return sites

def _generate_random(n, rng, radius=None):
    """Generate random compact cluster."""
    if radius is None:
        radius = n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = n * 50
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    # If we couldn't place enough, just add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _relax_one(positions, potential, rho):
    """Relax a single configuration."""
    try:
        energy, pos = relax(positions, potential, rho)
        return energy, pos
    except Exception:
        return float('inf'), positions

def construct(instance):
    """Find low-energy Morse cluster via diverse initializations + parallel relaxation."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse initial configurations
    inits = []
    
    # FCC-based
    try:
        fcc = _generate_fcc(n, rng)
        inits.append(("fcc", fcc))
    except:
        pass
    
    # HCP-based
    try:
        hcp = _generate_hcp(n, rng)
        inits.append(("hcp", hcp))
    except:
        pass
    
    # Icosahedral
    try:
        ico = _generate_icosahedral(n, rng)
        inits.append(("ico", ico))
    except:
        pass
    
    # Random compact (multiple variants)
    for i in range(4):
        try:
            rand = _generate_random(n, rng, radius=n**(1/3)*0.7*(1+i*0.1))
            inits.append((f"rand{i}", rand))
        except:
            pass
    
    # Additional icosahedral variants with different perturbations
    for i in range(2):
        try:
            ico2 = _generate_icosahedral(n, rng)
            inits.append((f"ico2_{i}", ico2))
        except:
            pass
    
    # Sort by initial energy (quick estimate)
    def init_energy(item):
        return _morse_energy(item[1], rho)
    
    inits.sort(key=init_energy)
    
    best_energy = float('inf')
    best_pos = None
    
    # Relax all initial configurations in parallel (up to 4 threads)
    max_workers = min(4, len(inits))
    
    # Process in batches to respect deadline
    for i in range(0, len(inits), max_workers):
        batch = inits[i:i+max_workers]
        if time.time() > deadline - 30:
            break
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [(name, executor.submit(_relax_one, pos, potential, rho)) for name, pos in batch]
            for name, fut in futures:
                try:
                    energy, pos = fut.result(timeout=60)
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = pos
                except:
                    pass
    
    if best_pos is None:
        # Fallback: return first valid initialization
        return inits[0][1]
    
    # Final high-precision polish: re-relax the best with tighter tolerance
    if time.time() < deadline - 10:
        try:
            energy2, pos2 = relax(best_pos, potential, rho)
            if energy2 < best_energy:
                best_energy = energy2
                best_pos = pos2
        except:
            pass
        
        # One more polish pass
        if time.time() < deadline - 5:
            try:
                energy3, pos3 = relax(best_pos, potential, rho)
                if energy3 < best_energy:
                    best_energy = energy3
                    best_pos = pos3
            except:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
