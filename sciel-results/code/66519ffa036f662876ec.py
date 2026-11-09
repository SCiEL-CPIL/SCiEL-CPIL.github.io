"""Improved program for Sutton-Chen cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_fcc_cluster(n, rng, spacing=1.0):
    """Generate a compact FCC-like cluster by taking the n closest points to origin in FCC lattice."""
    # FCC lattice vectors: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    # Generate points in a cube of lattice indices
    max_idx = int(np.ceil(n ** (1/3))) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    points.append(p)
    points = np.array(points)
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    # Take n closest, but ensure minimum distance
    selected = []
    min_dist = spacing * 0.95
    for idx in order:
        p = points[idx] * spacing
        if all(np.linalg.norm(p - q) >= min_dist for q in selected):
            selected.append(p)
        if len(selected) == n:
            break
    if len(selected) < n:
        # Fill remaining with random points
        while len(selected) < n:
            p = rng.uniform(-2, 2, 3) * spacing
            if all(np.linalg.norm(p - q) >= min_dist for q in selected):
                selected.append(p)
            else:
                break
    return np.array(selected)


def _generate_hcp_cluster(n, rng, spacing=1.0):
    """Generate a compact HCP-like cluster."""
    a = spacing
    c = np.sqrt(8/3) * a
    # HCP basis: layer A at z=0, layer B at z=c/2
    # In-plane: triangular lattice
    points = []
    max_r = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            # Layer A (z=0)
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3)/2) * j
            points.append(np.array([x, y, 0.0]))
            # Layer B (z=c/2)
            x_b = a * (i + 0.5 * j + 1/3)
            y_b = a * (np.sqrt(3)/2) * (j + 1/3)
            points.append(np.array([x_b, y_b, c/2]))
            # Layer C (z=c)
            x_c = a * (i + 0.5 * j + 2/3)
            y_c = a * (np.sqrt(3)/2) * (j + 2/3)
            points.append(np.array([x_c, y_c, c]))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    selected = []
    min_dist = spacing * 0.95
    for idx in order:
        p = points[idx]
        if all(np.linalg.norm(p - q) >= min_dist for q in selected):
            selected.append(p)
        if len(selected) == n:
            break
    if len(selected) < n:
        while len(selected) < n:
            p = rng.uniform(-3, 3, 3) * spacing
            if all(np.linalg.norm(p - q) >= min_dist for q in selected):
                selected.append(p)
            else:
                break
    return np.array(selected)


def _generate_random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.85 * spacing for q in points):
                points.append(p)
        attempts += 1
    # Fill if needed
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _generate_sphere_shell(n, rng, spacing=1.0):
    """Generate atoms on concentric shells (like icosahedral/ Mackay icosahedra)."""
    # Mackay icosahedra shells: 1, 12, 42, 100, 180, 280, ...
    # For arbitrary n, use nested shells
    shell_sizes = [1]
    total = 1
    shell_idx = 1
    while total < n:
        # Next shell size for icosahedra
        size = 12 * shell_idx * (shell_idx + 1)
        shell_sizes.append(size)
        total += size
        shell_idx += 1
    
    # Take from shells
    points = [np.zeros(3)]
    remaining = n - 1
    for s in shell_sizes[1:]:
        take = min(remaining, s)
        if take <= 0:
            break
        # Generate points on a sphere of radius proportional to shell index
        radius = spacing * 1.0 * np.sqrt(2 * s / 12)  # approximate
        # Use fibonacci sphere for uniform distribution
        phi = (1 + np.sqrt(5)) / 2
        for i in range(take):
            y = 1 - (2 * (i + 0.5)) / take
            r = np.sqrt(1 - y*y)
            theta = 2 * np.pi * i / phi
            p = radius * np.array([r * np.cos(theta), r * np.sin(theta), y])
            points.append(p)
        remaining -= take
        if remaining <= 0:
            break
    
    # Fill remaining
    while len(points) < n:
        p = rng.uniform(-1, 1, 3) * spacing * 2
        points.append(p)
    
    return np.array(points)


def _perturb(positions, rng, magnitude=0.1):
    """Add random perturbation to positions."""
    return positions + rng.normal(0, magnitude, positions.shape)


def construct(instance):
    """Multi-start optimization for Sutton-Chen cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    spacing = 1.0  # reduced units
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. FCC-like
    starts.append(_generate_fcc_cluster(size, rng, spacing))
    
    # 2. HCP-like
    starts.append(_generate_hcp_cluster(size, rng, spacing))
    
    # 3. Random compact
    for _ in range(3):
        starts.append(_generate_random_compact(size, rng, spacing))
    
    # 4. Shell/icosahedral
    starts.append(_generate_sphere_shell(size, rng, spacing))
    
    # 5. Perturbed FCC
    fcc_base = _generate_fcc_cluster(size, rng, spacing)
    for mag in [0.05, 0.1, 0.2]:
        starts.append(_perturb(fcc_base, rng, mag))
    
    # 6. Perturbed HCP
    hcp_base = _generate_hcp_cluster(size, rng, spacing)
    for mag in [0.05, 0.1, 0.2]:
        starts.append(_perturb(hcp_base, rng, mag))
    
    # 7. More random
    for _ in range(4):
        starts.append(_generate_random_compact(size, rng, spacing))
    
    # Relax each start and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, pos in enumerate(starts):
        # Ensure valid positions
        if not np.all(np.isfinite(pos)):
            continue
        # Check minimum distance
        min_d = np.inf
        for a in range(len(pos)):
            for b in range(a+1, len(pos)):
                d = np.linalg.norm(pos[a] - pos[b])
                if d < min_d:
                    min_d = d
        if min_d < 0.35:
            # Try to fix by relaxing anyway
            pass
        
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            continue
    
    # If we didn't get a good result, try a few more with different strategies
    if best_positions is None or best_energy > 1e6:
        for _ in range(5):
            pos = _generate_random_compact(size, rng, spacing)
            try:
                energy, relaxed_pos = relax(pos, potential, kind)
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                continue
    
    # Final refinement: run a few more local optimizations from the best
    if best_positions is not None:
        for _ in range(3):
            perturbed = _perturb(best_positions, rng, 0.03)
            try:
                energy, refined_pos = relax(perturbed, potential, kind)
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = refined_pos
            except Exception:
                continue
    
    return best_positions
# EVOLVE-BLOCK-END
