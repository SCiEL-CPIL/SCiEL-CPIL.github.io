"""Sutton-Chen transition-metal cluster global minima - DE-based global search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n, rng):
    """Generate a truncated FCC cluster with n atoms."""
    # FCC lattice vectors
    a = 1.0  # lattice constant in reduced units (equilibrium NN ~ 1.0)
    # Generate FCC points: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) + integer translations
    points = []
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if len(points) < n * 2:
                        points.append(p)
    # Sort by distance from origin and take n closest
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    # Center
    points -= points.mean(axis=0)
    # Add small random perturbation
    points += rng.normal(0, 0.01, points.shape)
    return points


def _hcp_positions(n, rng):
    """Generate a truncated HCP cluster with n atoms."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # HCP: two atoms per unit cell
                # Layer 1: (i, j, k)
                p1 = np.array([i + 0.5 * (1 if j % 2 == 0 else 0), j, k * c / a])
                # Layer 2: offset
                p2 = np.array([i + 0.5 + 0.5 * (1 if (j+1) % 2 == 0 else 0), j + 0.5, (k + 0.5) * c / a])
                if len(points) < n * 2:
                    points.append(p1)
                if len(points) < n * 2:
                    points.append(p2)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.01, points.shape)
    return points


def _icosahedral_positions(n, rng):
    """Generate an icosahedral cluster (Mackay or similar)."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts /= np.linalg.norm(verts[0])
    
    # Generate concentric icosahedral shells
    points = [np.array([0.0, 0.0, 0.0])]  # center
    shell_radii = [1.0, np.sqrt(3.0)/phi, 2.0]  # approximate radii
    for shell_r in shell_radii:
        for v in verts:
            p = shell_r * v
            if len(points) < n:
                points.append(p)
        # Also add edge midpoints for denser shells
        if len(points) < n:
            for i in range(12):
                for j in range(i+1, 12):
                    # Check if connected (roughly)
                    d = np.linalg.norm(verts[i] - verts[j])
                    if 0.9 < d < 1.1:
                        p = shell_r * (verts[i] + verts[j]) / 2
                        if len(points) < n:
                            points.append(p)
    
    points = np.array(points[:n])
    if len(points) < n:
        # Fill remaining with random points on sphere
        remaining = n - len(points)
        extra = rng.normal(0, 1, (remaining, 3))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        extra *= 1.0
        points = np.vstack([points, extra])
    
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.01, points.shape)
    return points


def _random_compact(n, rng):
    """Generate a random compact cluster."""
    radius = 0.8 * n ** (1.0/3.0)
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or all(np.linalg.norm(p - q) >= 0.7 for q in points):
                points.append(p)
        attempts += 1
    # If not enough points, add more randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    return points


def _relax_cluster(positions, potential, kind, maxiter=200):
    """Relax a cluster using L-BFGS-B."""
    try:
        energy, relaxed = relax(positions, potential, kind)
        return energy, relaxed
    except Exception:
        return float('inf'), positions


def _differential_evolution_step(pop, energies, rng, kind, potential, pop_size, 
                                  F=0.8, CR=0.9, maxiter=50):
    """One generation of differential evolution."""
    n = pop[0].shape[0]
    new_pop = list(pop)
    new_energies = list(energies)
    
    for i in range(pop_size):
        # Select three distinct individuals
        idxs = [i]
        while len(idxs) < 4:
            j = rng.integers(0, pop_size)
            if j not in idxs:
                idxs.append(j)
        
        a, b, c, d = idxs[1], idxs[2], idxs[3], i
        
        # Mutation
        mutant = pop[a] + F * (pop[b] - pop[c])
        
        # Crossover
        trial = np.zeros_like(pop[i])
        for j in range(n):
            for k in range(3):
                if rng.random() < CR or j == rng.integers(0, n):
                    trial[j, k] = mutant[j, k]
                else:
                    trial[j, k] = pop[i][j, k]
        
        # Relax the trial
        trial_energy, trial_relaxed = _relax_cluster(trial, potential, kind, maxiter)
        
        # Selection
        if trial_energy < new_energies[i]:
            new_pop[i] = trial_relaxed
            new_energies[i] = trial_energy
    
    return new_pop, new_energies


def construct(instance):
    """DE-based global search for Sutton-Chen cluster minima."""
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    start_time = time.time()
    deadline = 320.0  # Leave 10s margin before 330s kill
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial population
    pop_size = min(12, max(6, n // 5))
    
    # Templates
    templates = []
    templates.append(_fcc_positions(n, rng))
    templates.append(_hcp_positions(n, rng))
    templates.append(_icosahedral_positions(n, rng))
    for _ in range(pop_size - 3):
        templates.append(_random_compact(n, rng))
    
    # Relax all templates
    pop = []
    energies = []
    for t in templates:
        e, r = _relax_cluster(t, potential, kind)
        pop.append(r)
        energies.append(e)
    
    # Track best
    best_energy = min(energies)
    best_positions = pop[np.argmin(energies)]
    
    # Run DE iterations
    gen = 0
    while time.time() < deadline - 30.0:  # Leave 30s for final relaxation
        gen += 1
        pop, energies = _differential_evolution_step(
            pop, energies, rng, kind, potential, pop_size,
            F=0.8, CR=0.9, maxiter=50
        )
        
        # Update best
        if min(energies) < best_energy:
            best_energy = min(energies)
            best_positions = pop[np.argmin(energies)]
        
        # Occasionally add fresh random individuals to maintain diversity
        if gen % 5 == 0 and time.time() < deadline - 60.0:
            fresh = _random_compact(n, rng)
            fresh_e, fresh_r = _relax_cluster(fresh, potential, kind)
            worst_idx = np.argmax(energies)
            pop[worst_idx] = fresh_r
            energies[worst_idx] = fresh_e
    
    # Final relaxation of best
    final_e, final_r = _relax_cluster(best_positions, potential, kind, maxiter=500)
    if final_e < best_energy:
        best_energy = final_e
        best_positions = final_r
    
    # Validate
    if not np.all(np.isfinite(best_positions)):
        # Fallback to a simple random cluster
        best_positions = _random_compact(n, rng)
        best_energy, best_positions = _relax_cluster(best_positions, potential, kind)
    
    return best_positions
# EVOLVE-BLOCK-END
