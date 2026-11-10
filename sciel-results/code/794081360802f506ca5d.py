"""Dzugutov cluster global minima: polytetrahedral seed + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline management
_START_TIME = None
_DEADLINE = 315.0  # safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_polytetrahedral_seed(n, rng):
    """
    Generate a seed structure with polytetrahedral character.
    Uses a combination of icosahedral shells and random perturbations
    to create diverse starting points with local polytetrahedral order.
    """
    # Start with an icosahedral core
    phi = (1 + np.sqrt(5)) / 2
    icosahedron_verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    icosahedron_verts /= np.linalg.norm(icosahedron_verts[0])
    
    # Build shells: core (1), icosahedral shell (12), second shell (42), etc.
    # For polytetrahedral packing, use a BCC-like arrangement with icosahedral symmetry
    points = [np.zeros(3)]  # center atom
    
    # First shell: 12 atoms (icosahedron)
    shell1_scale = 1.0
    for v in icosahedron_verts:
        points.append(v * shell1_scale)
    
    # Second shell: 42 atoms (approximate)
    if n > 13:
        # Generate second shell positions using a dodecahedral/icosahedral pattern
        # Use the 30 edge midpoints of the icosahedron plus 12 face centers
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                dist = np.linalg.norm(icosahedron_verts[i] - icosahedron_verts[j])
                if dist < 1.5:  # close enough to be an edge
                    mid = (icosahedron_verts[i] + icosahedron_verts[j]) / 2
                    edges.append(mid)
        
        # Normalize and scale
        edges = np.array(edges)
        if len(edges) > 0:
            edges /= np.linalg.norm(edges[0])
            shell2_scale = 1.6
            for e in edges:
                if len(points) < n:
                    points.append(e * shell2_scale)
        
        # Add more atoms if needed by adding face centers
        if len(points) < n:
            # Add atoms at face centers of the icosahedron
            # Approximate by adding atoms in a spherical shell
            for _ in range(n - len(points)):
                theta = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(-1, 1)
                r = np.sqrt(1 - z * z)
                p = np.array([r * np.cos(theta), r * np.sin(theta), z])
                points.append(p * 2.2)
    
    # If we still need more atoms, add them in a random spherical shell
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        # Place in outer shell
        shell_radius = 2.5 + rng.uniform(0, 0.5)
        points.append(p * shell_radius)
    
    # If we have too many, truncate
    points = points[:n]
    
    # Apply small random perturbations to break symmetry and create diversity
    points = np.array(points)
    points += rng.normal(0, 0.1, points.shape)
    
    # Ensure minimum distance constraint
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < 0.5:
                # Push apart
                diff = points[i] - points[j]
                if np.linalg.norm(diff) < 1e-10:
                    diff = np.array([1, 0, 0])
                diff /= np.linalg.norm(diff)
                points[i] += diff * 0.26
                points[j] -= diff * 0.26
    
    return np.array(points)


def _generate_fcc_seed(n, rng):
    """Generate an FCC-like seed structure."""
    points = []
    # FCC unit cell basis
    basis = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ])
    
    # Generate atoms in a cubic grid
    side = int(np.ceil(n ** (1/3)))
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for b in basis:
                    if len(points) >= n:
                        break
                    p = np.array([ix, iy, iz]) * 1.1 + b * 1.1
                    points.append(p)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    # If not enough, add more
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * (side * 0.8))
    
    points = np.array(points[:n])
    points += rng.normal(0, 0.05, points.shape)
    return points


def _generate_random_compact(n, rng):
    """Generate a random compact cluster."""
    radius = 1.1 * n ** (1.0 / 3.0) * 0.7
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If not enough, add more without distance check
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * radius)
    
    return np.array(points[:n])


def _basin_hopping(positions, potential, rng, max_iters=50, step_scale=0.3):
    """
    Perform basin hopping: random displacement followed by local relaxation.
    Returns the best structure found.
    """
    best_positions = positions.copy()
    best_energy, _ = relax(positions, potential)
    
    for _ in range(max_iters):
        # Random displacement
        step = step_scale * rng.normal(0, 1, positions.shape)
        new_positions = positions + step
        
        # Relax
        energy, relaxed_positions = relax(new_positions, potential)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_positions.copy()
            positions = relaxed_positions.copy()
        else:
            # Accept with Metropolis-like criterion for diversity
            if rng.random() < np.exp(-0.5 * (energy - best_energy)):
                positions = relaxed_positions.copy()
            else:
                positions = positions  # keep previous
    
    return best_positions, best_energy


def construct(instance):
    """
    Find low-energy Dzugutov clusters using polytetrahedral seeds and basin hopping.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n * 7919)  # deterministic seed per n
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial structures
    seeds = []
    seeds.append(_generate_polytetrahedral_seed(n, rng))
    seeds.append(_generate_fcc_seed(n, rng))
    seeds.append(_generate_random_compact(n, rng))
    # Add a few more random seeds
    for _ in range(2):
        seeds.append(_generate_random_compact(n, rng))
    
    # Relax each seed and keep the best
    for seed in seeds:
        energy, positions = relax(seed, potential)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Basin hopping from the best structure
    time_remaining = _get_time_remaining()
    if time_remaining > 30:
        # Do basin hopping
        max_iters = min(100, int(time_remaining / 2))
        step_scale = 0.3
        for iteration in range(max_iters):
            if _get_time_remaining() < 20:
                break
            positions, energy = _basin_hopping(
                best_positions, potential, rng, 
                max_iters=1, 
                step_scale=step_scale
            )
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                # Occasionally reduce step size for refinement
                if iteration % 20 == 19 and step_scale > 0.1:
                    step_scale *= 0.8
    
    # Final relaxation
    if best_positions is not None:
        final_energy, final_positions = relax(best_positions, potential)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    if best_positions is None:
        # Fallback: return a simple linear arrangement
        best_positions = np.array([[i * 1.1, 0, 0] for i in range(n)])
    
    return best_positions
# EVOLVE-BLOCK-END
