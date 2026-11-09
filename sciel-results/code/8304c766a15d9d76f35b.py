"""Dzugutov cluster global minima via basin hopping with diverse polytetrahedral seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill


def _energy(positions, n):
    """Compute Dzugutov energy for n atoms."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Only compute for pairs i < j
    iu = np.triu_indices(n, 1)
    r = dist[iu]
    V = np.zeros_like(r)
    mask_a = r < a
    mask_b = (r >= a) & (r < b)
    V[mask_a] = A * (r[mask_a]**(-m) - B) * np.exp(c / (r[mask_a] - a))
    V[mask_b] = B * np.exp(d / (r[mask_b] - b))
    return np.sum(V)


def _icosahedron_vertices(scale=1.0):
    """Return 12 vertices of a regular icosahedron scaled so edge length ~ scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    d = np.linalg.norm(verts[0] - verts[4])
    verts = verts * (scale / d)
    return verts


def _generate_icosahedral_cluster(n, scale=1.1):
    """Generate an n-atom cluster based on icosahedral geometry."""
    # 12 vertices of icosahedron
    ico = _icosahedron_vertices(scale)
    # 6 face centers (for icosahedron, each face center)
    # Actually, let's use a simpler approach: start with icosahedron and add atoms
    # For n > 12, add atoms on faces or in interstitial positions
    
    points = list(ico)
    
    if n <= 12:
        return np.array(points[:n])
    
    # Add atoms progressively
    rng = np.random.default_rng(42)
    remaining = n - 12
    
    # Try to place additional atoms near existing ones
    for _ in range(remaining):
        # Pick a random existing atom and offset it
        idx = rng.integers(0, len(points))
        base = points[idx]
        # Random small displacement
        offset = rng.normal(0, 0.3 * scale, 3)
        new_p = base + offset
        # Check minimum distance
        valid = True
        for p in points:
            if np.linalg.norm(new_p - p) < 0.5:
                valid = False
                break
        if valid:
            points.append(new_p)
        else:
            # Try a larger displacement
            for _ in range(10):
                offset = rng.normal(0, 0.5 * scale, 3)
                new_p = base + offset
                valid = True
                for p in points:
                    if np.linalg.norm(new_p - p) < 0.5:
                        valid = False
                        break
                if valid:
                    points.append(new_p)
                    break
    
    return np.array(points[:n])


def _generate_fcc_cluster(n, scale=1.1):
    """Generate an n-atom cluster based on FCC geometry."""
    rng = np.random.default_rng(43)
    points = []
    a = scale * 0.707  # FCC lattice constant for nearest-neighbor distance ~ scale
    
    # Generate FCC points in a cube
    max_coord = int(np.ceil((n * (a**3))**(1/3))) + 1
    candidates = []
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # FCC positions
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    r = np.linalg.norm(p)
                    if r < 3.0 * scale:
                        candidates.append(p)
    
    # Select n points with minimum distance constraint
    candidates = np.array(candidates)
    selected = []
    for p in candidates:
        if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
            selected.append(p)
            if len(selected) == n:
                break
    
    if len(selected) < n:
        # Add random points
        while len(selected) < n:
            p = rng.normal(0, scale, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
                selected.append(p)
    
    return np.array(selected[:n])


def _generate_random_cluster(n, scale=1.1, seed=44):
    """Generate a random n-atom cluster."""
    rng = np.random.default_rng(seed)
    points = []
    radius = scale * n ** (1/3) * 0.8
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add them with larger radius
    while len(points) < n:
        p = rng.normal(0, radius * 1.5, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        else:
            # Force placement with small perturbation
            base = points[rng.integers(0, len(points))]
            p = base + rng.normal(0, 0.6, 3)
            points.append(p)
    
    return np.array(points[:n])


def _basin_hopping(positions, n, rng, max_steps=50, step_size=0.5):
    """Perform basin hopping starting from initial positions."""
    current = positions.copy()
    current_energy = _energy(current, n)
    best = current.copy()
    best_energy = current_energy
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Random perturbation
        perturbed = current + rng.normal(0, step_size, current.shape)
        
        # Relax the perturbed structure
        try:
            energy, relaxed = relax(perturbed, 'dzugutov', None)
        except:
            energy = _energy(perturbed, n)
            relaxed = perturbed
        
        # Accept or reject based on Metropolis criterion
        dE = energy - current_energy
        if dE < 0 or rng.random() < np.exp(-dE / 0.5):
            current = relaxed
            current_energy = energy
            if energy < best_energy:
                best = relaxed.copy()
                best_energy = energy
        
        # Adapt step size
        if step % 10 == 0:
            step_size = max(0.1, min(1.0, step_size * 0.9))
    
    return best, best_energy


def construct(instance):
    """Find low-energy Dzugutov cluster using basin hopping with diverse seeds."""
    size = instance["n"]
    potential = instance["potential"]
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial structures
    seed_structures = []
    
    # Icosahedral-based
    ico = _generate_icosahedral_cluster(size, scale=1.1)
    if ico.shape[0] == size:
        seed_structures.append(ico)
    
    # FCC-based
    fcc = _generate_fcc_cluster(size, scale=1.1)
    if fcc.shape[0] == size:
        seed_structures.append(fcc)
    
    # Random clusters with different seeds
    for seed in [100, 200, 300]:
        rand = _generate_random_cluster(size, scale=1.1, seed=seed)
        if rand.shape[0] == size:
            seed_structures.append(rand)
    
    # Run basin hopping from each seed
    best_positions = None
    best_energy = np.inf
    
    for i, seed_pos in enumerate(seed_structures):
        if time.time() > _DEADLINE - 10:
            break
        
        # Quick relaxation first
        try:
            energy, relaxed = relax(seed_pos, potential, None)
        except:
            energy = _energy(seed_pos, size)
            relaxed = seed_pos
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed
        
        # Basin hopping from relaxed structure
        bh_pos, bh_energy = _basin_hopping(relaxed, size, rng, max_steps=30, step_size=0.4)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    # If no valid structure found, return a random one
    if best_positions is None:
        best_positions = _generate_random_cluster(size, scale=1.1, seed=999)
    
    return best_positions
# EVOLVE-BLOCK-END
