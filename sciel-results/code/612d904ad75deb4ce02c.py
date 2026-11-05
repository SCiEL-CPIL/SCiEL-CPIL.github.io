"""Dzugutov cluster global minima search with polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

# Dzugutov potential parameters
A_const = 5.82
B_const = 1.28
a_const = 1.87
b_const = 1.94
c_const = 1.10
d_const = 0.27
m_const = 16.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy exactly."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a_const:
                energy += A_const * (r ** (-m_const) - B_const) * np.exp(c_const / (r - a_const))
            elif r < b_const:
                energy += B_const * np.exp(d_const / (r - b_const))
    return energy


def _generate_icosahedron(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _generate_mackay_icosahedron(n):
    """
    Generate a Mackay icosahedral cluster of size n.
    Structure: 1 + 12 + 12 + 12 + ...
    For n > 13, we place atoms on the vertices of the icosahedron,
    then on the edges, then on the faces if necessary, scaling the shell.
    However, for Dzugutov, the 'shells' are often just additional icosahedra
    with larger radii, but the Mackay construction (filling the icosahedron)
    is a better geometric prior for the *topology* of the neighbors.
    
    A simpler and often effective approach for Dzugutov is concentric icosahedra
    with radii optimized for the potential minimum (~1.0 for first shell).
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    # Start with center
    pts = [np.zeros(3)]
    
    # Shell 1: 12 atoms
    if n > 1:
        shell1 = _generate_icosahedron(1.0)
        pts.extend(shell1)
    
    # If n > 13, add more shells
    # The Dzugutov potential minimum is around r=1.0 for the 13-atom cluster.
    # For larger clusters, the outer shells expand.
    # Empirical radii for Dzugutov global minima (approx):
    # 13: R=1
    # 25: R1=1, R2=1.8
    # 37: R1=1, R2=1.8, R3=2.6
    # 49: R1=1, R2=1.8, R3=2.6, R4=3.4
    
    radii = [1.0, 1.8, 2.6, 3.4, 4.2]
    shell_idx = 1 # We already did shell 0 (center) and shell 1 (12 atoms)
    current_count = 13
    
    while current_count < n:
        # Get next shell radius
        if shell_idx < len(radii):
            r = radii[shell_idx]
        else:
            r = radii[-1] + 0.8 * (shell_idx - len(radii) + 1)
            
        shell = _generate_icosahedron(r)
        
        # Add as many as needed
        needed = n - current_count
        if needed >= 12:
            pts.extend(shell)
            current_count += 12
        else:
            # Add partial shell
            # To maintain symmetry as much as possible, pick the first 'needed' vertices
            # This breaks symmetry but relaxation will fix it
            pts.extend(shell[:needed])
            current_count += needed
            
        shell_idx += 1
        
    return np.array(pts[:n])


def _grow_cluster(start_positions, n_target, rng):
    """
    Greedy growth: Start with a valid cluster (e.g., 13 atoms) and add atoms
    one by one at the position that minimizes the energy of the new atom
    with respect to the existing cluster. Then relax.
    """
    pts = list(start_positions)
    current_n = len(pts)
    
    # Pre-compute energies for speed? No, just simple loop for small n.
    # We will try a few candidate positions for each new atom.
    
    while current_n < n_target:
        best_pos = None
        best_e = float('inf')
        
        # Generate candidate positions on a sphere around the center
        # Radius roughly matching the current shell
        # Estimate current radius
        dists = np.linalg.norm(pts, axis=1)
        avg_r = np.mean(dists)
        # Try a range of radii
        for r_test in [avg_r * 0.8, avg_r, avg_r * 1.2, avg_r * 1.5]:
            # Generate random points on this sphere
            for _ in range(50):
                # Random direction
                u = rng.random(3)
                u /= np.linalg.norm(u)
                pos = u * r_test
                
                # Check min distance to existing atoms
                dists_to_existing = np.linalg.norm(np.array(pts) - pos, axis=1)
                if np.min(dists_to_existing) < 0.5:
                    continue
                
                # Calculate energy of this new atom with existing ones
                e_new = 0.0
                for p in pts:
                    r = np.linalg.norm(p - pos)
                    if r < 1e-12:
                        continue
                    if r < a_const:
                        e_new += A_const * (r ** (-m_const) - B_const) * np.exp(c_const / (r - a_const))
                    elif r < b_const:
                        e_new += B_const * np.exp(d_const / (r - b_const))
                    
                    if e_new > best_e: # Early termination
                        break
                
                if e_new < best_e:
                    best_e = e_new
                    best_pos = pos
        
        if best_pos is None:
            # Fallback: random position
            best_pos = rng.normal(0, avg_r, 3)
            
        pts.append(best_pos)
        current_n += 1
        
    return np.array(pts)


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_energy = float('inf')
    best_positions = None
    
    rng = np.random.default_rng(size)
    
    candidates = []
    
    # 1. Mackay Icosahedral Shells
    # Generate a few variations with slight radial distortions
    for r1_scale in [0.95, 1.0, 1.05]:
        for r2_scale in [1.7, 1.8, 1.9]:
            # Build manually to allow scaling
            pts = [np.zeros(3)]
            shell1 = _generate_icosahedron(1.0 * r1_scale)
            pts.extend(shell1)
            
            current_count = 13
            shell_idx = 1
            radii_base = [1.0, 1.8, 2.6, 3.4, 4.2]
            
            while current_count < size:
                if shell_idx < len(radii_base):
                    r = radii_base[shell_idx]
                    if shell_idx == 1:
                        r *= r2_scale
                    elif shell_idx == 2:
                        r *= r2_scale * 0.95 # Slight variation
                    elif shell_idx == 3:
                        r *= r2_scale * 0.90
                else:
                    r = radii_base[-1] + 0.8 * (shell_idx - len(radii_base) + 1)
                
                shell = _generate_icosahedron(r)
                needed = size - current_count
                if needed >= 12:
                    pts.extend(shell)
                    current_count += 12
                else:
                    pts.extend(shell[:needed])
                    current_count += needed
                shell_idx += 1
                
            candidates.append(np.array(pts[:size]))
            
    # 2. Greedy Growth from 13-atom Icosahedron
    # This is often very effective for finding the specific global minimum topology
    if size > 13:
        start_13 = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
        # Run a few greedy growths with different random seeds for candidate positions
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            grown = _grow_cluster(start_13, size, rng)
            candidates.append(grown)
            
    # 3. Random Compact Structures
    # A few random structures to ensure we don't miss a non-icosahedral minimum (unlikely but possible)
    for _ in range(2):
        R = 1.0 * (size / 4.0) ** (1/3)
        pts = rng.uniform(-R, R, (size, 3))
        candidates.append(pts)
        
    # Relax each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
            
        try:
            # Use relax
            energy, positions = relax(cand, potential, None)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            pass
            
    # If we have time left, do some perturbation-based search on the best structure
    if best_positions is not None and time.time() < deadline - 30:
        # Try to improve the best structure by adding noise and re-relaxing
        # Use larger noise to escape local minima
        for i in range(10):
            if time.time() > deadline:
                break
            # Scale noise with cluster size
            noise_scale = 0.1 * (size / 13.0) ** 0.5
            noise = rng.normal(0, noise_scale, best_positions.shape)
            perturbed = best_positions + noise
            try:
                energy, positions = relax(perturbed, potential, None)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except:
                pass

    if best_positions is None:
        # Fallback
        best_positions = candidates[0]
        
    return best_positions
# EVOLVE-BLOCK-END
