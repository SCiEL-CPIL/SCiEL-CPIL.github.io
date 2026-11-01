import numpy as np
import time
import random

N = 55

# EVOLVE-BLOCK-START
import numpy as np

# Assume science_chemistry is available in the environment
try:
    import science_chemistry
    RELAX_AVAILABLE = True
except ImportError:
    RELAX_AVAILABLE = False

def _lj_energy(pos):
    """Calculate Lennard-Jones energy."""
    n = pos.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = np.sum((pos[i] - pos[j]) ** 2)
            if r2 < 1e-12:
                return 1e10
            r_inv2 = 1.0 / r2
            r_inv6 = r_inv2 ** 3
            energy += 4.0 * (r_inv6 ** 2 - r_inv6)
    return energy

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts, dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_icosahedral_seed(n_atoms=55, noise_scale=0.1):
    """
    Generate a seed with icosahedral symmetry:
    1 center, 12 icosahedron vertices, and remaining atoms randomly placed
    in the region of the second shell.
    """
    pos = np.zeros((n_atoms, 3))
    # Atom 0 is center
    # Atoms 1-12 are icosahedron vertices
    ico_verts = _icosahedron_vertices()
    # Scale icosahedron vertices to typical LJ shell radius (~1.1)
    pos[1:13] = ico_verts * 1.1
    
    # Remaining atoms (42)
    # Place them randomly in a spherical shell between r=1.5 and r=2.0
    # This encourages the second shell formation
    remaining = n_atoms - 13
    if remaining > 0:
        # Random directions
        random_dirs = np.random.normal(0, 1, (remaining, 3))
        random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
        
        # Random radii
        random_radii = np.random.uniform(1.5, 2.0, remaining)
        
        pos[13:] = random_dirs * random_radii[:, np.newaxis]
        
        # Add small noise to break perfect symmetry and help escape local minima
        if noise_scale > 0:
            pos += np.random.normal(0, noise_scale, pos.shape)
            
    return pos

def _relax_and_evaluate(pos, max_iters=1000):
    """
    Relax the structure using science_chemistry.relax if available,
    otherwise use a simple local search or just return current energy.
    """
    if RELAX_AVAILABLE:
        try:
            relaxed_pos = science_chemistry.relax(pos)
            energy = _lj_energy(relaxed_pos)
            return relaxed_pos, energy
        except Exception:
            pass
            
    # Fallback: Simple local relaxation if relax is not available or fails
    # This is a basic Nelder-Mead-like or gradient-free step
    # Given the time limit, we might just return the input if relax fails
    return pos, _lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_energy = float('inf')
    best_pos = None
    
    # Initial structures to try
    seeds = []
    
    # 1. Current Seed: Cubic Lattice
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    cubic_seed = np.array(points[:N], dtype=float) * 1.12
    seeds.append(cubic_seed)
    
    # 2. Icosahedral Seed (Low Noise)
    seeds.append(_generate_icosahedral_seed(noise_scale=0.05))
    
    # 3. Icosahedral Seed (Medium Noise)
    seeds.append(_generate_icosahedral_seed(noise_scale=0.2))
    
    # 4. Random Spherical Seed
    random_dirs = np.random.normal(0, 1, (N, 3))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    random_radii = np.random.uniform(1.0, 2.5, N)
    seeds.append(random_dirs * random_radii[:, np.newaxis])
    
    # 5. Perturbed Cubic
    seeds.append(cubic_seed + np.random.normal(0, 0.2, cubic_seed.shape))

    # Iterate through seeds, relaxing each
    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
            
        # Ensure minimum distance constraint is not violated immediately (0.5 apart)
        # The relax function should handle this, but we can check.
        # If atoms are too close, relax might fail.
        
        try:
            relaxed_pos, energy = _relax_and_evaluate(seed)
            
            # Verify validity: atoms at least 0.5 apart
            valid = True
            for k in range(N):
                for l in range(k + 1, N):
                    if np.linalg.norm(relaxed_pos[k] - relaxed_pos[l]) < 0.5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
                
        except Exception as e:
            # If relaxation fails, skip
            continue
            
    # If no valid result found, return the cubic seed as fallback
    if best_pos is None:
        return cubic_seed
        
    return best_pos
# EVOLVE-BLOCK-END
