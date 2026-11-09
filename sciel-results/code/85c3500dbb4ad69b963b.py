"""Dzugutov cluster global minima (polytetrahedral model metal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants for Dzugutov potential
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(pos):
    """Compute the Dzugutov potential energy for a set of positions."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Mask out self-distances
    np.fill_diagonal(dists, np.inf)
    
    energy = 0.0
    # Only need to sum over i < j
    for i in range(n):
        for j in range(i + 1, n):
            r = dists[i, j]
            if r < 1e-10:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy

def _generate_icosahedron_shells(n):
    """
    Generate a cluster based on icosahedral shells.
    Mackay icosahedron shells: 1, 12, 42, 92, ...
    For n=55, it's 1+12+42 = 55.
    For other n, we approximate by taking the closest shell configuration.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12 atoms)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit length
    norm = np.linalg.norm(verts[0])
    verts = verts / norm
    
    # Shell 0: center (1 atom)
    points = [np.zeros(3)]
    
    # Shell 1: 12 vertices (total 13)
    if n > 1:
        points.extend(verts.tolist())
    
    # Shell 2: 42 atoms (total 55)
    # These are the centers of the 20 faces and 30 edges of the icosahedron,
    # plus some additional positions. Actually, the second shell of a Mackay icosahedron
    # consists of 42 atoms.
    # We can generate them by finding the midpoints of edges and face centers,
    # but a more robust way is to use the known coordinates or generate them
    # by placing atoms in the gaps.
    
    # For simplicity, let's generate points in the second shell by sampling
    # directions that correspond to the 42 positions.
    # The 42 positions are the vertices of a cuboctahedron-like structure? No.
    # They are the centers of the 20 triangular faces and the 30 edges? 20+30=50, not 42.
    # Actually, the Mackay icosahedron second shell has 42 atoms.
    # Let's just use a radial expansion with specific angles if possible,
    # or fall back to random packing in the shell if we can't easily generate them.
    
    # A simpler approach for general n:
    # If n <= 13, return the first n points.
    if n <= 13:
        return np.array(points[:n])
    
    # For n > 13, we need to add atoms in the second shell.
    # The second shell radius is roughly sqrt(2) * R1? Or 2 * R1?
    # In a Mackay icosahedron, the second shell atoms are at a distance of
    # about 1.73 * R1 (where R1 is the distance to the first shell).
    # Let's try R2 = 1.73
    
    R2 = 1.73
    rng = np.random.default_rng(42)
    
    # Generate candidate points in the second shell
    # We want 42 points. Let's try to place them in the gaps.
    # The gaps are near the face centers and edge midpoints.
    # Face centers: 20. Edge midpoints: 30. Total 50.
    # But we only need 42.
    # Let's generate 50 candidates and pick the best 42 (or n-13).
    
    candidates = []
    
    # Face centers
    # We need to find the faces of the icosahedron.
    # This is complex. Let's just sample random points in the shell
    # and use a greedy algorithm to place them with minimum distance.
    
    # Number of atoms to add in the second shell
    num_to_add = min(n - 13, 42)  # Max 42 in the second shell
    
    current_points = np.array(points)
    min_dist = 0.8  # Minimum distance between atoms
    
    # Generate a large number of random candidates in the shell
    num_candidates = 200
    shell_candidates = []
    for _ in range(num_candidates):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        p = v * R2
        shell_candidates.append(p)
    
    shell_candidates = np.array(shell_candidates)
    
    # Greedily select the best candidates
    selected = []
    for p in shell_candidates:
        if len(selected) >= num_to_add:
            break
        # Check distance against all current points (including already selected)
        dists = np.linalg.norm(current_points - p, axis=1)
        if np.min(dists) >= min_dist:
            # Also check against already selected in this shell
            if selected:
                sel_dists = np.linalg.norm(np.array(selected) - p, axis=1)
                if np.min(sel_dists) < min_dist:
                    continue
            selected.append(p)
    
    # If we didn't get enough, relax the distance constraint
    if len(selected) < num_to_add:
        min_dist = 0.6
        selected = []
        for p in shell_candidates:
            if len(selected) >= num_to_add:
                break
            dists = np.linalg.norm(current_points - p, axis=1)
            if np.min(dists) >= min_dist:
                if selected:
                    sel_dists = np.linalg.norm(np.array(selected) - p, axis=1)
                    if np.min(sel_dists) < min_dist:
                        continue
                selected.append(p)
    
    # If still not enough, just add random points
    while len(selected) < num_to_add:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        p = v * R2
        selected.append(p)
    
    points.extend(selected)
    
    # If n > 55, we need a third shell
    if n > 55:
        num_to_add = n - 55
        R3 = 2.6  # Approximate third shell radius
        current_points = np.array(points)
        selected = []
        min_dist = 0.8
        
        num_candidates = 200
        shell_candidates = []
        for _ in range(num_candidates):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            p = v * R3
            shell_candidates.append(p)
        
        shell_candidates = np.array(shell_candidates)
        
        for p in shell_candidates:
            if len(selected) >= num_to_add:
                break
            dists = np.linalg.norm(current_points - p, axis=1)
            if np.min(dists) >= min_dist:
                if selected:
                    sel_dists = np.linalg.norm(np.array(selected) - p, axis=1)
                    if np.min(sel_dists) < min_dist:
                        continue
                selected.append(p)
        
        points.extend(selected)
    
    return np.array(points[:n])

def _generate_fcc(n):
    """Generate a truncated FCC cluster."""
    rng = np.random.default_rng(123 + n)
    points = []
    # FCC lattice basis vectors
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]
    
    # Generate points in a grid
    N = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                for b in basis:
                    p = np.array([i, j, k]) + np.array(b)
                    candidates.append(p)
    
    candidates = np.array(candidates)
    # Sort by distance from center
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    candidates = candidates[indices]
    
    # Select closest n points
    points = candidates[:n]
    
    # Center it
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    return points

def _generate_random_packing(n, seed):
    """Generate a random compact packing."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.1 * n ** (1.0 / 3.0)
    min_dist = 0.8
    attempts = 0
    max_attempts = n * 50
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
        
    # If we didn't get enough, just fill with random points (might overlap, relax will fix)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _basin_hopping(pos, potential, parameter, max_steps=10, timeout=None):
    """
    Simple basin hopping: perturb and relax.
    """
    rng = np.random.default_rng(1000 + hash(tuple(pos[0])) % 10000)
    current_energy, current_pos = relax(pos, potential, parameter)
    
    for step in range(max_steps):
        if timeout and time.time() > timeout:
            break
        
        # Perturbation
        sigma = 0.15 * np.exp(-step * 0.1)
        noise = rng.normal(0, sigma, size=current_pos.shape)
        perturbed_pos = current_pos + noise
        
        try:
            energy, new_pos = relax(perturbed_pos, potential, parameter)
            if energy < current_energy:
                current_energy = energy
                current_pos = new_pos
        except Exception:
            continue
    
    return current_energy, current_pos

def construct(instance):
    """
    Find the lowest-energy structure for a Dzugutov cluster.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    timeout = 295.0  # Safety margin before 330s kill
    
    # Generate initial candidates
    candidates = []
    
    # 1. Icosahedral template (most important for Dzugutov)
    try:
        p_ico = _generate_icosahedron_shells(size)
        candidates.append(p_ico)
    except Exception:
        pass
        
    # 2. FCC template
    try:
        p_fcc = _generate_fcc(size)
        candidates.append(p_fcc)
    except Exception:
        pass
        
    # 3. Random packings (a few seeds)
    for seed in [1, 2, 3, 4, 5]:
        try:
            p_rand = _generate_random_packing(size, seed)
            candidates.append(p_rand)
        except Exception:
            pass

    best_energy = np.inf
    best_pos = None
    
    # Relax all candidates
    for i, pos in enumerate(candidates):
        if time.time() - start_time > timeout - 10:
            break
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
        except Exception:
            continue
            
    # Refinement: Basin Hopping style
    # Take the best structure, perturb, and relax again.
    if best_pos is not None:
        # Use more basin hopping steps if we have time
        remaining_time = timeout - (time.time() - start_time)
        max_refine_steps = min(20, int(remaining_time / 2))  # Assume ~2s per step
        
        energy, pos = _basin_hopping(best_pos, potential, parameter, 
                                      max_steps=max_refine_steps,
                                      timeout=start_time + timeout - 5)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    if best_pos is None:
        # Fallback: Return one of the generated candidates if relax failed entirely
        if candidates:
            return candidates[0]
        else:
            # Last resort: linear chain (invalid for energy, but valid geometry?)
            # Task says "atoms must be at least 0.5 apart".
            return np.array([np.array([i, 0, 0]) for i in range(size)])

    return best_pos
# EVOLVE-BLOCK-END
