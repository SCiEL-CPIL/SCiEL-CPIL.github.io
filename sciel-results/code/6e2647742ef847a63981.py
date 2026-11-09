"""Improved program for Sutton-Chen cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_fcc_cluster(n, rng, spacing=1.0):
    """Generate a compact FCC-like cluster by taking the n closest points to origin in FCC lattice."""
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    max_idx = int(np.ceil(n ** (1/3))) + 3
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    r = np.linalg.norm(p)
                    points.append((r, p))
    points.sort(key=lambda x: x[0])
    # Take n points, scale to appropriate spacing
    selected = [p for _, p in points[:n]]
    selected = np.array(selected) * spacing
    # Center the cluster
    selected -= selected.mean(axis=0)
    return selected


def _generate_hcp_cluster(n, rng, spacing=1.0):
    """Generate a compact HCP-like cluster."""
    a = spacing
    c = a * np.sqrt(2.0 / 3.0) * 2
    points = []
    layers = int(np.ceil(n ** (1/3))) + 3
    for layer in range(-layers, layers + 1):
        z = layer * c / 2
        # AB stacking
        if layer % 2 == 0:
            offset = np.array([0, 0, 0])
        else:
            offset = np.array([a/2, a/(2*np.sqrt(3)), 0])
        for i in range(-layers, layers + 1):
            for j in range(-layers, layers + 1):
                p = np.array([i * a + (j % 2) * a/2 + offset[0],
                             j * a * np.sqrt(3)/2 + offset[1],
                             z + offset[2]])
                r = np.linalg.norm(p)
                points.append((r, p))
    points.sort(key=lambda x: x[0])
    selected = np.array([p for _, p in points[:n]])
    selected -= selected.mean(axis=0)
    return selected


def _generate_random_compact(n, rng, spacing=0.72):
    """Generate a random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    min_dist = 0.6 * spacing
    attempts = 0
    max_attempts = 50000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all points, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _generate_sphere_shell(n, rng, spacing=0.72):
    """Generate points on spherical shells."""
    points = []
    shells = int(np.ceil(n ** (1/3)))
    for s in range(1, shells + 1):
        shell_radius = spacing * s * 0.8
        # Fibonacci sphere for uniform distribution
        k = int(np.pi * (5 ** 0.5))
        i = len(points)
        while len(points) < n and i < n + s:
            t = (i + 0.5) / n
            z = 1 - 2 * t
            r = np.sqrt(1 - z*z)
            phi = k * i
            p = shell_radius * np.array([r * np.cos(phi), r * np.sin(phi), z])
            points.append(p)
            i += 1
    if len(points) > n:
        points = points[:n]
    while len(points) < n:
        p = rng.uniform(-spacing, spacing, 3)
        points.append(p)
    return np.array(points)


def _perturb(positions, rng, scale):
    """Apply random perturbation to positions."""
    return positions + rng.normal(0, scale, positions.shape)


def _is_valid(positions):
    """Check if positions are finite and no two atoms are closer than 0.35."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < 0.35:
                return False
    return True


def construct(instance):
    """Best of multiple strategies with basin hopping for Sutton-Chen cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    # Set deadline: 300s total, leave 15s safety margin
    deadline = time.time() + 285
    rng = np.random.default_rng(size)
    
    # Generate diverse initial structures
    initial_structures = []
    
    # FCC-based
    initial_structures.append(_generate_fcc_cluster(size, rng, spacing=0.72))
    
    # HCP-based
    initial_structures.append(_generate_hcp_cluster(size, rng, spacing=0.72))
    
    # Random compact
    initial_structures.append(_generate_random_compact(size, rng, spacing=0.72))
    
    # Sphere shell
    initial_structures.append(_generate_sphere_shell(size, rng, spacing=0.72))
    
    # Additional random structures
    for _ in range(3):
        initial_structures.append(_generate_random_compact(size, rng, spacing=0.72))
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Local relaxation of each initial structure
    for i, init in enumerate(initial_structures):
        if time.time() > deadline - 30:
            break
        try:
            energy, positions = relax(init, potential, kind)
            if _is_valid(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Phase 2: Basin hopping from best structure
    if best_positions is not None and time.time() < deadline - 60:
        current_positions = best_positions.copy()
        current_energy = best_energy
        temp = 0.5
        base_scale = 0.3
        max_iterations = 50
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 15:
                break
            
            # Perturb
            scale = base_scale * (1 + 0.5 * np.sin(iteration * 0.3))
            perturbed = _perturb(current_positions, rng, scale)
            
            # Relax
            try:
                energy, positions = relax(perturbed, potential, kind)
                if not _is_valid(positions):
                    continue
            except Exception:
                continue
            
            # Metropolis acceptance
            delta_e = energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / (temp + 1e-10)):
                current_positions = positions
                current_energy = energy
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
                    # Cool down when we find improvement
                    temp *= 0.8
            else:
                temp *= 1.05  # Heat up when rejected
            
            # Adaptive temperature
            temp = max(0.1, min(2.0, temp))
    
    # Phase 3: Additional basin hopping from other good structures
    if time.time() < deadline - 30:
        # Sort initial structures by their relaxed energies (we didn't track all, so re-relax a few)
        for i in [0, 1, 2]:
            if time.time() > deadline - 15:
                break
            init = initial_structures[i]
            try:
                energy, positions = relax(init, potential, kind)
                if not _is_valid(positions):
                    continue
            except Exception:
                continue
            
            # Short basin hopping from this structure
            current_positions = positions.copy()
            current_energy = energy
            temp = 0.3
            for _ in range(15):
                if time.time() > deadline - 10:
                    break
                scale = 0.25
                perturbed = _perturb(current_positions, rng, scale)
                try:
                    e, pos = relax(perturbed, potential, kind)
                    if not _is_valid(pos):
                        continue
                except Exception:
                    continue
                
                delta_e = e - current_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / (temp + 1e-10)):
                    current_positions = pos
                    current_energy = e
                    if e < best_energy:
                        best_energy = e
                        best_positions = pos
                temp = max(0.1, min(1.5, temp * 1.02))
    
    # Final validation
    if best_positions is None:
        # Fallback: return a simple compact structure
        best_positions = _generate_fcc_cluster(size, rng, spacing=0.72)
    
    # Ensure validity
    if not _is_valid(best_positions):
        # Try to fix by spreading atoms
        for _ in range(100):
            min_dist = np.inf
            for i in range(len(best_positions)):
                for j in range(i+1, len(best_positions)):
                    d = np.linalg.norm(best_positions[i] - best_positions[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist >= 0.35:
                break
            # Push apart the closest pair
            for i in range(len(best_positions)):
                for j in range(i+1, len(best_positions)):
                    d = np.linalg.norm(best_positions[i] - best_positions[j])
                    if d < 0.35:
                        direction = (best_positions[i] - best_positions[j]) / (d + 1e-10)
                        shift = (0.35 - d + 0.01) / 2
                        best_positions[i] += direction * shift
                        best_positions[j] -= direction * shift
    
    return best_positions
# EVOLVE-BLOCK-END
