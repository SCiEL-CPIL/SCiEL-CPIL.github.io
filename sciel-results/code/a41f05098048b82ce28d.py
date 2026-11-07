"""
Strategy: Structure-Driven Basin Hopping with Explicit Geometric Motifs.

This approach differs fundamentally from the previous parallel multi-start strategy by:
1. Abandoning generic random/soft-sphere initializations in favor of explicit,
   physically motivated structural fragments (FCC, HCP, BCC shells).
2. Using a single, serial Basin Hopping loop that is more predictable and less prone
   to GIL contention issues compared to 4 parallel threads.
3. Implementing an adaptive step-size mechanism that aggressively explores when
   stuck and refines when finding improvements.
4. Prioritizing diverse structural families to ensure the search samples different
   regions of the configuration space (close-packed vs. body-centered).
"""

import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 290.0  # Safety margin before 300s hard kill
_START_TIME = None

def _init_global():
    global _START_TIME
    _START_TIME = time.time()

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n):
    """Generate a compact FCC-like cluster."""
    rng = np.random.default_rng(42)
    points = []
    # Generate FCC lattice points within a sphere
    # a = 1.0 (lattice constant)
    # Nearest neighbor distance ~ a/sqrt(2) ~ 0.707
    # We want min distance > 0.35, so this is safe.
    
    max_radius = 1.5 * n ** (1/3)
    attempts = 0
    max_attempts = 5000
    
    while len(points) < n and attempts < max_attempts:
        # Sample random point in sphere
        r = max_radius * rng.random() ** (1/3)
        theta = rng.random() * 2 * np.pi
        phi = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        
        # Snap to nearest FCC lattice point
        # FCC lattice: (i, j, k) where i+j+k is even, scaled by a/sqrt(2)
        # Actually, standard FCC primitive vectors:
        # a1 = (0, 0.5, 0.5), a2 = (0.5, 0, 0.5), a3 = (0.5, 0.5, 0)
        # Lattice points are integer combinations of these.
        # Equivalent to: (x, y, z) where x,y,z are half-integers or integers, and x+y+z is integer.
        # Let's just use a simple cubic grid and filter for FCC-like symmetry or just use a dense random start 
        # and let the relaxer do the heavy lifting, but biasing towards compactness.
        
        # A simpler robust FCC generator:
        # Use a list of known FCC coordinates for small n, or generate a sphere of FCC points.
        # For now, let's generate a dense random cluster and rely on the fact that FCC is the global min for many sizes.
        # To make it "FCC-like", we can initialize with a BCC or FCC supercell and cut.
        
        # Let's try a different tactic: Generate a spherical shell of points and relax.
        # But the prompt asks for "different strategy". The current one uses "packed sphere".
        # Let's implement actual FCC fragment generation.
        
        pass # This function is being re-implemented below in a cleaner way.
    
    # Fallback: Random compact
    R = 1.2 * n ** (1/3)
    return rng.uniform(-R, R, (n, 3))

def _generate_fcc_cluster(n):
    """Generate a cluster of n atoms from an FCC lattice."""
    rng = np.random.default_rng(123)
    points = []
    # FCC lattice points: (i, j, k) where i,j,k are integers, and i+j+k is even.
    # Scale factor: a = 1.0. Distance between neighbors = 1/sqrt(2) ~ 0.707.
    # This is > 0.35, so valid.
    
    # Generate a list of candidate FCC points sorted by distance from origin
    candidates = []
    L = int(np.ceil(1.5 * n ** (1/3))) + 2
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2))
                    r = np.linalg.norm(p)
                    candidates.append((r, p))
    
    candidates.sort(key=lambda x: x[0])
    
    # Select n closest points
    for _, p in candidates:
        if len(points) >= n:
            break
        points.append(p)
        
    if len(points) < n:
        # If we didn't get enough (unlikely for reasonable n), fill with random
        remaining = n - len(points)
        R = 2.0 * n ** (1/3)
        extra = rng.uniform(-R, R, (remaining, 3))
        points.extend(extra.tolist())
        
    return np.array(points[:n])

def _generate_hcp_cluster(n):
    """Generate a cluster of n atoms from an HCP lattice."""
    rng = np.random.default_rng(456)
    points = []
    # HCP lattice: Two interpenetrating simple hexagonal lattices.
    # Basis: (0,0,0) and (1/3, 2/3, 1/2) in hexagonal coordinates?
    # Let's use a simpler approximation: FCC is often close to HCP.
    # For a distinct strategy, we can use a BCC-like or just a different random seed with a different packing.
    # Let's implement a simple hexagonal close packed layer stacking.
    
    # Layer 1: Triangular lattice
    # Layer 2: Offset triangular lattice
    # Layer 3: Same as Layer 1
    
    a = 1.0 # lattice constant
    h = a * np.sqrt(2/3) # height between layers for ideal HCP
    
    candidates = []
    L = int(np.ceil(1.5 * n ** (1/3))) + 2
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # Hexagonal lattice points
                x = a * (i + j/2)
                y = a * (j * np.sqrt(3) / 2)
                z = h * k
                p = np.array([x, y, z])
                r = np.linalg.norm(p)
                candidates.append((r, p))
                
                # Second sublattice (offset)
                # Offset by (1/3, 2/3) in hex coords?
                # In Cartesian: x_offset = a/3, y_offset = a*sqrt(3)/6, z_offset = h/2
                x2 = x + a/3
                y2 = y + a*np.sqrt(3)/6
                z2 = z + h/2
                p2 = np.array([x2, y2, z2])
                r2 = np.linalg.norm(p2)
                candidates.append((r2, p2))
                
    candidates.sort(key=lambda x: x[0])
    
    for _, p in candidates:
        if len(points) >= n:
            break
        points.append(p)
        
    if len(points) < n:
        remaining = n - len(points)
        R = 2.0 * n ** (1/3)
        extra = rng.uniform(-R, R, (remaining, 3))
        points.extend(extra.tolist())
        
    return np.array(points[:n])

def _generate_bcc_cluster(n):
    """Generate a cluster of n atoms from a BCC lattice."""
    rng = np.random.default_rng(789)
    points = []
    # BCC lattice: Simple cubic + center at (0.5, 0.5, 0.5)
    a = 1.0
    # Nearest neighbor distance in BCC is a*sqrt(3)/2 ~ 0.866.
    
    candidates = []
    L = int(np.ceil(1.5 * n ** (1/3))) + 2
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                p1 = np.array([i, j, k]) * a
                r1 = np.linalg.norm(p1)
                candidates.append((r1, p1))
                
                p2 = (np.array([i, j, k]) + 0.5) * a
                r2 = np.linalg.norm(p2)
                candidates.append((r2, p2))
                
    candidates.sort(key=lambda x: x[0])
    
    for _, p in candidates:
        if len(points) >= n:
            break
        points.append(p)
        
    if len(points) < n:
        remaining = n - len(points)
        R = 2.0 * n ** (1/3)
        extra = rng.uniform(-R, R, (remaining, 3))
        points.extend(extra.tolist())
        
    return np.array(points[:n])

def _min_distance(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    if n < 2:
        return float('inf')
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def construct(instance):
    global _START_TIME
    _init_global()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Define the set of initial structures to try
    # We will cycle through these in the basin hopping loop
    # 1. FCC
    # 2. HCP
    # 3. BCC
    # 4. Random Compact
    # 5. Random Loose
    
    def _random_compact(seed):
        rng = np.random.default_rng(seed)
        R = 1.2 * n ** (1/3)
        return rng.uniform(-R, R, (n, 3))
        
    def _random_loose(seed):
        rng = np.random.default_rng(seed)
        R = 2.0 * n ** (1/3)
        return rng.uniform(-R, R, (n, 3))
        
    initial_structures = [
        _generate_fcc_cluster(n),
        _generate_hcp_cluster(n),
        _generate_bcc_cluster(n),
        _random_compact(1),
        _random_loose(2)
    ]
    
    # Ensure all initial structures are valid (min dist > 0.35)
    # If not, scale them up
    for i in range(len(initial_structures)):
        md = _min_distance(initial_structures[i])
        if md < 0.35:
            # Scale up to ensure validity
            scale = 0.35 / md * 1.1
            initial_structures[i] = initial_structures[i] * scale
            
    best_energy = float('inf')
    best_positions = None
    
    # Basin Hopping Loop
    # We start with the FCC structure as it's often the global min or close to it
    current_pos = initial_structures[0].copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    
    if not np.isfinite(current_energy):
        # Fallback to random if FCC fails
        current_pos = initial_structures[3].copy()
        current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
        
    if current_energy < best_energy:
        best_energy = current_energy
        best_positions = current_pos.copy()
        
    step_size = 0.3
    max_iterations = 5000 # High cap, but time-limited
    
    # We will periodically switch the "base" structure to one of the other initial structures
    # to ensure we explore different basins
    base_structure_idx = 0
    switch_interval = 50 # Switch base structure every 50 iterations
    
    for i in range(max_iterations):
        if _time_left() < 5:
            break
            
        # Perturb current position
        noise = np.random.normal(0, step_size, (n, 3))
        perturbed = current_pos + noise
        
        try:
            energy, positions = relax(perturbed, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(energy):
            step_size *= 1.2
            continue
            
        min_d = _min_distance(positions)
        if min_d < 0.35:
            step_size *= 1.2
            continue
            
        if energy < current_energy:
            current_energy = energy
            current_pos = positions
            step_size *= 0.95 # Reduce step size on success
        else:
            # Metropolis acceptance? Or just reject?
            # Basin Hopping usually accepts all moves to escape local minima, 
            # but keeps the best. Let's accept with some probability to explore.
            # For simplicity and speed, let's just reject and increase step size.
            step_size *= 1.05
            step_size = min(step_size, 1.0)
            
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
            
        # Periodically reset current state to a different base structure
        if i % switch_interval == 0 and i > 0:
            base_structure_idx = (base_structure_idx + 1) % len(initial_structures)
            # Perturb the new base structure slightly to start hopping from there
            base_pos = initial_structures[base_structure_idx].copy()
            noise = np.random.normal(0, 0.1, (n, 3))
            current_pos = base_pos + noise
            current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
            
            if not np.isfinite(current_energy):
                current_pos = initial_structures[0].copy()
                current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
                
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_pos.copy()
                
    # Final check
    if best_positions is None or not np.isfinite(best_energy):
        # Fallback
        best_positions = initial_structures[0]
        best_energy, _ = sutton_chen_energy_gradient(best_positions, kind)
        
    # Ensure final validity
    if best_positions is not None:
        md = _min_distance(best_positions)
        if md < 0.35:
            scale = 0.35 / md * 1.1
            best_positions = best_positions * scale
            
    return best_positions
