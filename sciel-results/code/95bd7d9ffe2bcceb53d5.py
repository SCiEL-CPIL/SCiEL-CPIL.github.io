"""Rewritten program: icosahedral seed generation + basin hopping for Dzugutov clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax

# Dzugutov potential parameters
DZUGUTOV_PARAMS = {
    'A': 5.82, 'B': 1.28, 'a': 1.87, 'b': 1.94, 'c': 1.10, 'd': 0.27, 'm': 16
}

# Icosahedron vertices (12 vertices)
def _icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    # The 12 vertices of a regular icosahedron
    verts = np.array([
        [ 1,  1,  0], [-1,  1,  0], [ 1, -1,  0], [-1, -1,  0],
        [ 0,  1,  1], [ 0, -1,  1], [ 0,  1, -1], [ 0, -1, -1],
        [ 1,  0,  1], [-1,  0,  1], [ 1,  0, -1], [-1,  0, -1],
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

_ICO_VERTS = _icosahedron_vertices()

def _generate_icosahedral_seed(n, rng):
    """
    Generate an icosahedrally-symmetric seed structure with n atoms.
    Uses concentric icosahedral shells with some atoms removed to match n.
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    # Build shells: center + icosahedral shells
    # Shell 0: 1 atom (center)
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 30 atoms (icosahedron edges midpoints, scaled)
    # Shell 3: 60 atoms (icosahedron face centers, scaled)
    
    shells = []
    
    # Shell 0: center
    shells.append(np.array([[0.0, 0.0, 0.0]]))
    
    # Shell 1: 12 icosahedron vertices at radius r1
    r1 = 1.0
    shell1 = _ICO_VERTS * r1
    shells.append(shell1)
    
    # Shell 2: 30 atoms - midpoints of icosahedron edges, scaled
    # Icosahedron has 30 edges. We can generate edge midpoints.
    # For simplicity, use a slightly larger icosahedron with 30 points
    # Actually, let's use the 30 edge midpoints of the icosahedron
    # The icosahedron vertices are at unit distance. Edge midpoints are at distance ~0.951
    # But for a shell, we want them at a larger radius.
    
    # Alternative: use icosahedral symmetry groups
    # Shell 2 (30 atoms): can be generated as icosahedron vertices scaled + rotated
    # Let's use a different approach: generate points on icosahedral shells
    
    # For shell 2, use the 30 edge midpoints of the icosahedron, but scaled to be at a larger radius
    # Actually, the standard icosahedral shell of 30 atoms corresponds to the vertices of a
    # truncated icosahedron or similar. Let's use a simpler approach.
    
    # Generate 30 points: take icosahedron vertices and create 5-fold symmetric arrangements
    # Simpler: use the fact that 30 = 5 * 6, so 6 rings of 5 atoms each
    
    # For now, let's use a practical approach: generate icosahedral shells by scaling
    # and using known icosahedral point sets
    
    # Shell 2: 30 atoms at radius r2
    # Use icosahedral symmetry: 30 points can be arranged as 5 squares rotated
    # Or use the 30 edge midpoints of the icosahedron scaled up
    r2 = 1.618  # golden ratio, typical for icosahedral shells
    
    # Generate 30 points using icosahedral symmetry
    # The 30 edge midpoints of an icosahedron
    # Icosahedron edges connect vertices that are not opposite
    # For each pair of vertices that form an edge, compute midpoint
    
    # Icosahedron adjacency: each vertex connects to 5 others
    # Let's compute edge midpoints
    verts = _ICO_VERTS
    n_verts = len(verts)
    edge_midpoints = []
    for i in range(n_verts):
        for j in range(i+1, n_verts):
            dist = np.linalg.norm(verts[i] - verts[j])
            # Icosahedron edge length is 2/sqrt(phi^2+1) * something
            # Actually, for unit icosahedron, edge length = 2/sqrt(phi^2+1)
            # Let's just check if they're close (edge)
            if 0.9 < dist < 1.2:  # approximate edge length for unit icosahedron
                mid = (verts[i] + verts[j]) / 2.0
                edge_midpoints.append(mid)
    
    if len(edge_midpoints) > 0:
        edge_midpoints = np.array(edge_midpoints)
        # Normalize to unit sphere then scale
        norms = np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
        edge_midpoints = edge_midpoints / norms
        shell2 = edge_midpoints * r2
        shells.append(shell2)
    
    # Shell 3: 60 atoms at radius r3
    r3 = 2.236  # sqrt(5)
    
    # For 60 atoms, use icosahedral face centers
    # Icosahedron has 20 faces, each with 3 vertices
    # Face centers give 20 points, not 60
    # 60 = 2 * 30, so use two sets of 30
    # Or use the 60 vertices of a dodecahedron-like structure
    
    # Simpler: use icosahedral symmetry to generate 60 points
    # 60 = 12 * 5, so 5 points per icosahedron vertex direction
    # Or use the 60 edge midpoints of a dodecahedron
    
    # For practical purposes, let's generate 60 points by taking the 30 edge midpoints
    # and their antipodes, or use a different method
    
    # Actually, for the clusters we care about (38-60), we may not need all shells
    # Let's build the full structure and then select n atoms
    
    # Build all shell atoms
    all_atoms = []
    for shell in shells:
        all_atoms.append(shell)
    
    # Concatenate
    all_atoms = np.vstack(all_atoms)
    
    # If we have more atoms than needed, remove some from the outer shell
    if len(all_atoms) > n:
        # Remove atoms from the outermost shell
        # Shuffle to break symmetry slightly
        indices = np.arange(len(all_atoms))
        rng.shuffle(indices)
        all_atoms = all_atoms[indices[:n]]
    elif len(all_atoms) < n:
        # Add random atoms near the surface
        while len(all_atoms) < n:
            # Add atom at random position on a sphere of radius r_outer
            r_outer = r3 if r3 > 0 else 2.0
            theta = rng.uniform(0, 2*np.pi)
            phi = rng.uniform(0, np.pi)
            p = np.array([
                r_outer * np.sin(phi) * np.cos(theta),
                r_outer * np.sin(phi) * np.sin(theta),
                r_outer * np.cos(phi)
            ])
            # Check minimum distance
            if len(all_atoms) > 0:
                dists = np.linalg.norm(all_atoms - p, axis=1)
                if np.min(dists) >= 0.5:
                    all_atoms = np.vstack([all_atoms, p])
            else:
                all_atoms = np.array([p])
    
    return all_atoms[:n]

def _generate_fcc_seed(n, rng):
    """Generate an FCC-like seed structure."""
    # FCC lattice points within a sphere
    a = 1.0  # lattice constant
    points = []
    # Generate FCC points in a cube and select those within radius
    max_coord = int(np.ceil(n ** (1/3) * a / 2)) + 2
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i*a + dx*a, j*a + dy*a, k*a + dz*a])
                    if np.linalg.norm(p) <= n ** (1/3) * a * 0.7:
                        points.append(p)
    
    points = np.array(points)
    if len(points) > n:
        indices = np.arange(len(points))
        rng.shuffle(indices)
        points = points[indices[:n]]
    elif len(points) < n:
        while len(points) < n:
            p = rng.uniform(-3, 3, 3)
            if len(points) > 0:
                dists = np.linalg.norm(points - p, axis=1)
                if np.min(dists) >= 0.5:
                    points = np.vstack([points, p])
            else:
                points = np.array([p])
    return points[:n]

def _basin_hop(positions, potential, parameter, rng, max_steps=50, step_size=0.5):
    """
    Basin hopping: perturb, relax, accept/reject.
    Returns best energy and positions found.
    """
    current_energy, current_pos = relax(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(max_steps):
        # Perturb
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + perturbation
        
        # Relax
        try:
            new_energy, new_pos = relax(perturbed, potential, parameter)
        except Exception:
            continue
        
        # Accept if better, or with Metropolis criterion
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.5):
            current_energy = new_energy
            current_pos = new_pos
    
    return best_energy, best_pos

def _solve_one_n(n, deadline, n_seeds=6, max_bh_steps=30):
    """Solve for a specific n, returning best positions."""
    potential = 'dzugutov'
    parameter = None  # default parameters
    rng = np.random.default_rng(n * 12345 + 67890)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse seeds
    seeds = []
    
    # Icosahedral seeds with different perturbations
    for i in range(n_seeds // 2):
        seed = _generate_icosahedral_seed(n, rng)
        # Add small perturbation for diversity
        seed = seed + rng.normal(0, 0.1, size=seed.shape)
        seeds.append(seed)
    
    # FCC seeds
    for i in range(n_seeds // 2):
        seed = _generate_fcc_seed(n, rng)
        seed = seed + rng.normal(0, 0.1, size=seed.shape)
        seeds.append(seed)
    
    # Random compact seed
    radius = 1.1 * n ** (1.0/3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    seeds.append(np.array(points))
    
    # Process each seed with basin hopping
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        
        try:
            # First relax the seed
            energy, pos = relax(seed, potential, parameter)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
            
            # Basin hopping
            bh_energy, bh_pos = _basin_hop(pos, potential, parameter, rng, 
                                           max_steps=max_bh_steps, step_size=0.5)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
                
        except Exception as e:
            continue
    
    if best_pos is None:
        # Fallback: return a valid random structure
        radius = 1.1 * n ** (1.0/3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
        best_pos = np.array(points)
    
    return best_pos

def construct(instance):
    """Construct lowest-energy structure for given cluster size."""
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin before 330s limit
    
    n = instance['n']
    
    # Solve for this n
    positions = _solve_one_n(n, deadline)
    
    # Ensure minimum distance constraint
    # If any pair is too close, try to fix
    for _ in range(10):
        min_dist = np.inf
        min_pair = None
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= 0.5:
            break
        # Move the two atoms apart
        i, j = min_pair
        direction = positions[i] - positions[j]
        if np.linalg.norm(direction) < 1e-10:
            direction = np.array([1.0, 0.0, 0.0])
        direction = direction / np.linalg.norm(direction)
        shift = (0.5 - min_dist) / 2.0 + 0.01
        positions[i] += direction * shift
        positions[j] -= direction * shift
    
    return positions
# EVOLVE-BLOCK-END
