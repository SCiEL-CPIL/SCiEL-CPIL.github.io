"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices():
    """Return normalized icosahedron vertices (12 points)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def _icosahedron_edges():
    """Return edge midpoint directions (30 edges, normalized)."""
    verts = _icosahedron_vertices()
    dists = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    edge_pairs = [(dists[k][1], dists[k][2]) for k in range(30)]
    edge_dirs = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        edge_dirs.append(mid / np.linalg.norm(mid))
    return np.array(edge_dirs)


def _icosahedron_face_centers():
    """Return 20 face center directions (normalized)."""
    verts = _icosahedron_vertices()
    faces = [
        [0, 1, 4], [0, 4, 8], [0, 8, 10], [0, 10, 6], [0, 6, 1],
        [1, 6, 5], [1, 5, 7], [1, 7, 2], [1, 2, 0],
        [2, 7, 5], [2, 5, 9], [2, 9, 3], [2, 3, 1],
        [3, 9, 4], [3, 4, 8], [3, 8, 11], [3, 11, 7], [3, 7, 2],
        [4, 9, 11], [4, 11, 8],
    ]
    centers = []
    for f in faces:
        c = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        centers.append(c / np.linalg.norm(c))
    return np.array(centers)


def _build_icosahedral_cluster(n, seed=0, radii_scale=1.0, include_center=True, jitter=0.0):
    """Build a cluster with icosahedral core + surface atoms at high-symmetry sites."""
    rng = np.random.default_rng(seed)
    positions = []
    
    # Core icosahedron: 12 atoms
    core_radius = 0.72 * radii_scale
    core = _icosahedron_vertices() * core_radius
    if jitter > 0:
        core += rng.normal(0, jitter, core.shape)
    positions.extend(core)
    
    # Add center atom if n >= 13
    if n >= 13 and include_center:
        positions.append([0.0, 0.0, 0.0])
    
    n_remaining = n - len(positions)
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Second shell: 30 edge-midpoint positions
    edge_dirs = _icosahedron_edges()
    shell2_radius = 1.05 * radii_scale
    shell2 = edge_dirs * shell2_radius
    if jitter > 0:
        shell2 += rng.normal(0, jitter, shell2.shape)
        
    n_add2 = min(n_remaining, 30)
    positions.extend(shell2[:n_add2])
    n_remaining -= n_add2
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Third shell: 20 face centers
    face_dirs = _icosahedron_face_centers()
    shell3_radius = 1.35 * radii_scale
    shell3 = face_dirs * shell3_radius
    if jitter > 0:
        shell3 += rng.normal(0, jitter, shell3.shape)

    n_add3 = min(n_remaining, 20)
    positions.extend(shell3[:n_add3])
    n_remaining -= n_add3
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Fourth shell: Fibonacci sphere for remaining atoms
    if n_remaining > 0:
        indices = np.arange(n_remaining)
        phi = np.arccos(1 - 2 * (indices + 0.5) / n_remaining)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        shell4_radius = 1.55 * radii_scale
        shell4 = np.column_stack([
            shell4_radius * np.sin(phi) * np.cos(theta),
            shell4_radius * np.sin(phi) * np.sin(theta),
            shell4_radius * np.cos(phi)
        ])
        if jitter > 0:
            shell4 += rng.normal(0, jitter, shell4.shape)
        positions.extend(shell4)
    
    return np.array(positions[:n])


def _build_fcc_fragment(n, seed=0):
    """Build FCC lattice fragment."""
    rng = np.random.default_rng(seed)
    a = 0.72
    basis = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ]
    
    side = int(np.ceil((n / 4) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)[:n]
    positions = candidates[indices]
    positions -= np.mean(positions, axis=0)
    return positions


def _build_hcp_fragment(n, seed=0):
    """Build HCP lattice fragment."""
    rng = np.random.default_rng(seed)
    a = 0.72
    c = a * np.sqrt(8.0/3.0)
    layer_spacing = c / 2.0
    
    positions = []
    n_layers = int(np.ceil(np.sqrt(n / 3.0))) + 2
    
    for layer in range(n_layers):
        z = layer * layer_spacing
        offset = np.array([0, 0, 0]) if layer % 2 == 0 else np.array([a/2, a/(2*np.sqrt(3)), 0])
        
        max_r = int(np.ceil(np.sqrt(n / (3 * np.pi / (a**2 * 0.866)))))
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                x = a * (i + j / 2) + offset[0]
                y = a * np.sqrt(3) / 2 * j + offset[1]
                p = np.array([x, y, z])
                if len(positions) < n:
                    positions.append(p)
                else:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _build_random_cluster(n, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _validate(positions, min_dist=0.35):
    """Check if positions are valid (finite, no overlaps)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        dists = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
        if np.any(dists < min_dist):
            return False
    return True


def _relax_and_validate(positions, potential, kind):
    """Relax positions and return (energy, valid_positions)."""
    try:
        energy, pos = relax(positions, potential, kind)
        if _validate(pos):
            return energy, pos
    except Exception:
        pass
    return np.inf, positions


def _ga_search(instance, deadline):
    """Genetic Algorithm search for global minimum."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    rng = np.random.default_rng(42)
    
    pop_size = 12
    generations = 15
    
    # Initialize population
    population = []
    # Mix of templates
    templates = [
        _build_icosahedral_cluster(size, seed=0, radii_scale=1.0),
        _build_icosahedral_cluster(size, seed=1, radii_scale=1.1),
        _build_fcc_fragment(size, seed=0),
        _build_fcc_fragment(size, seed=1),
        _build_hcp_fragment(size, seed=0),
        _build_hcp_fragment(size, seed=1),
        _build_random_cluster(size, seed=0),
        _build_random_cluster(size, seed=1),
        _build_random_cluster(size, seed=2),
        _build_random_cluster(size, seed=3),
        _build_icosahedral_cluster(size, seed=2, radii_scale=0.9),
        _build_fcc_fragment(size, seed=2),
    ]
    
    for i in range(pop_size):
        if i < len(templates):
            pos = templates[i].copy()
        else:
            pos = _build_random_cluster(size, seed=i)
        if not _validate(pos):
            pos = _build_random_cluster(size, seed=100+i)
        population.append(pos)
    
    # Initial relaxation
    def eval_pop(pop):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_and_validate, p, potential, kind): i for i, p in enumerate(pop)}
            energies = [np.inf] * len(pop)
            relaxed_pos = [p.copy() for p in pop]
            for future in concurrent.futures.as_completed(futures):
                if time.time() > deadline:
                    break
                idx = futures[future]
                try:
                    e, pos = future.result()
                    energies[idx] = e
                    relaxed_pos[idx] = pos
                except Exception:
                    pass
            return energies, relaxed_pos

    energies, population = eval_pop(population)
    
    best_energy = np.min(energies)
    best_pos = population[np.argmin(energies)].copy()
    
    for gen in range(generations):
        if time.time() > deadline:
            break
            
        # Selection: Tournament
        def select_tournament(pop, energies, k=3):
            idxs = rng.integers(0, len(pop), k)
            best_idx = idxs[np.argmin([energies[i] for i in idxs])]
            return pop[best_idx].copy()
            
        # Create offspring
        new_population = []
        for _ in range(pop_size):
            parent1 = select_tournament(population, energies)
            parent2 = select_tournament(population, energies)
            
            # Crossover: Swap random subset of atoms
            child = parent1.copy()
            n_swap = rng.integers(3, size // 2)
            swap_indices = rng.choice(size, n_swap, replace=False)
            child[swap_indices] = parent2[swap_indices]
            
            # Mutation: Add noise
            noise = rng.normal(0, 0.2, child.shape)
            child += noise
            
            if not _validate(child):
                # If invalid, fall back to parent1 with smaller noise
                child = parent1 + rng.normal(0, 0.1, parent1.shape)
                if not _validate(child):
                    child = parent1.copy()
            
            new_population.append(child)
        
        # Relax offspring
        new_energies, new_population = eval_pop(new_population)
        
        # Combine and select best for next generation
        all_population = population + new_population
        all_energies = energies + new_energies
        
        # Sort by energy
        sorted_indices = np.argsort(all_energies)
        population = [all_population[i].copy() for i in sorted_indices[:pop_size]]
        energies = [all_energies[i] for i in sorted_indices[:pop_size]]
        
        # Update global best
        gen_best = np.min(energies)
        if gen_best < best_energy:
            best_energy = gen_best
            best_pos = population[np.argmin(energies)].copy()
            
    return best_pos


def construct(instance):
    """Structure prediction using Genetic Algorithm for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Run GA
    try:
        best_pos = _ga_search(instance, deadline)
    except Exception:
        # Fallback
        best_pos = _build_icosahedral_cluster(size, seed=0)
        try:
            _, best_pos = relax(best_pos, potential, kind)
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
