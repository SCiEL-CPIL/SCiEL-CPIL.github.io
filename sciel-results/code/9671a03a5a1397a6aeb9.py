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
    if np.any(dists[np.triu_indices(n, k=1)] < min_dist):
        return False
    return True

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 2
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
    max_r = int(np.ceil(n ** (1/3))) + 2
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
    """Generate an icosahedral cluster using Mackay icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    
    # Generate icosahedral points
    pts = [np.zeros(3)]
    
    # 12 vertices of icosahedron
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    if n > 1:
        pts.extend(v[:n-1])
    
    if n > 13:
        # Add second shell (42 atoms) - Mackay icosahedron
        # Second shell positions are at midpoints of edges and face centers
        # For simplicity, add points at radius ~1.8 in icosahedral directions
        
        # Generate 42 points: 30 edge midpoints + 12 face centers
        # Edge midpoints: average of pairs of adjacent vertices
        edge_mids = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(v[i] - v[j])
                if 0.9 < d < 1.1:  # Adjacent vertices
                    edge_mids.append((v[i] + v[j]) / 2)
        
        # Face centers: average of triplets
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(v[i] - v[j])
                    d2 = np.linalg.norm(v[j] - v[k])
                    d3 = np.linalg.norm(v[k] - v[i])
                    if all(0.9 < d < 1.1 for d in [d1, d2, d3]):
                        fc = (v[i] + v[j] + v[k]) / 3
                        face_centers.append(fc)
        
        # Scale to appropriate radius
        second_shell = []
        for p in edge_mids[:30]:
            second_shell.append(p * 1.8)
        for p in face_centers[:12]:
            second_shell.append(p * 1.8)
        
        needed = n - len(pts)
        pts.extend(second_shell[:needed])
    
    return np.array(pts[:n])

def _basin_hopping_step(pos, kind, rng):
    """Perform one basin hopping step: perturb, relax, accept/reject."""
    n = len(pos)
    
    # Perturb with Gaussian noise
    sigma = 0.3
    perturbed = pos + rng.normal(0, sigma, pos.shape)
    
    # Relax the perturbed structure
    try:
        energy, relaxed = relax(perturbed, 'sutton_chen', kind)
    except Exception:
        return pos, np.inf
    
    if not np.isfinite(energy):
        return pos, np.inf
    
    if not _is_valid(relaxed):
        return pos, np.inf
    
    return relaxed, energy

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Generate diverse starting structures
    generators = [_generate_fcc, _generate_bcc, _generate_icosahedron]
    
    # Try each generator as a starting point for basin hopping
    for gen in generators:
        if _time_left() < 15.0:
            break
        
        try:
            base = gen(n)
        except Exception:
            continue
        
        # Relax the base structure
        try:
            energy, relaxed = relax(base, 'sutton_chen', kind)
        except Exception:
            continue
        
        if not np.isfinite(energy) or not _is_valid(relaxed):
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
        
        # Perform basin hopping from this structure
        current_energy = energy
        current_pos = relaxed.copy()
        
        # Adaptive temperature for basin hopping
        T = 0.5
        max_bh_steps = 50
        
        for step in range(max_bh_steps):
            if _time_left() < 10.0:
                break
            
            # Perform basin hopping step
            new_pos, new_energy = _basin_hopping_step(current_pos, kind, rng)
            
            if not np.isfinite(new_energy):
                continue
            
            # Metropolis acceptance
            if new_energy < current_energy:
                current_pos = new_pos
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            else:
                # Accept with probability exp(-(E_new - E_old) / T)
                delta_E = new_energy - current_energy
                if rng.random() < np.exp(-delta_E / T):
                    current_pos = new_pos
                    current_energy = new_energy
            
            # Slowly decrease temperature
            T *= 0.95
        
        # Increase perturbation size for next generator
        # (already handled in _basin_hopping_step)
    
    # If no valid position found, return a default valid one
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            # Fallback: try to fix by spreading atoms
            best_pos = rng.uniform(-2, 2, (n, 3))
            for _ in range(10):
                if _is_valid(best_pos):
                    break
                best_pos = rng.uniform(-2, 2, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
