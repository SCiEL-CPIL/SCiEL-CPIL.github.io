import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

def _dzugutov_energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    # Mask for i<j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = dist[mask]
    energy = 0.0
    for r in dists:
        if r < a:
            energy += A * (r**(-m) - B) * np.exp(c / (r - a))
        elif r < b:
            energy += B * np.exp(d / (r - b))
    return energy

def _generate_random_cluster(n, seed, min_dist=0.5):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    # Estimate radius based on n
    radius = 2.0 * n ** (1/3)
    points = []
    attempts = 0
    max_attempts = 50000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
            attempts = 0
        else:
            attempts += 1
    if len(points) == n:
        return np.array(points)
    return None

def _basin_hopping(positions, potential, rng, max_iterations=50, noise_scale=0.3, 
                   accept_threshold=0.01, time_limit=20):
    """
    Basin hopping: perturb, relax, accept/reject based on energy.
    Returns (best_energy, best_positions).
    """
    start_time = time.time()
    current_energy, current_pos = relax(positions, potential)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb
        noise = rng.normal(0, noise_scale, current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        try:
            new_energy, new_pos = relax(perturbed, potential)
        except Exception:
            continue
        
        # Accept if better, or with probability exp(-delta/T)
        delta = new_energy - current_energy
        if delta < 0:
            accept = True
        else:
            # Simple Metropolis-like acceptance with cooling
            T = max(0.1, accept_threshold * (1.0 - it / max_iterations))
            accept = rng.random() < np.exp(-delta / T)
        
        if accept:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
    
    return best_energy, best_pos

def construct(instance):
    n = instance['n']
    potential = instance['potential']
    
    start_time = time.time()
    time_budget = 280  # Leave 20s margin
    
    # Generate multiple diverse starting points
    rng = np.random.default_rng(42)
    
    # Strategy: Multi-start basin hopping with different initial conditions
    # and different noise scales to explore the rugged landscape
    
    best_energy = np.inf
    best_positions = None
    
    # Number of independent basin hopping runs based on time budget
    # Each run takes ~5-15s, so we can do ~15-30 runs
    n_runs = 20
    
    # Different initial conditions:
    # 1. Random clusters with varying density
    # 2. Random clusters with different noise scales
    
    for run_idx in range(n_runs):
        if time.time() - start_time > time_budget:
            break
        
        # Vary the initial cluster generation
        seed = 1000 + run_idx * 17
        
        # Vary the minimum distance to get different packing densities
        min_dist = 0.5 + 0.1 * (run_idx % 3)
        
        # Vary the radius
        radius_scale = 0.8 + 0.2 * (run_idx % 4)
        
        # Generate initial cluster
        positions = _generate_random_cluster(n, seed, min_dist=min_dist)
        if positions is None:
            continue
        
        # Run basin hopping with varying noise scales
        noise_scales = [0.2, 0.3, 0.5, 0.8]
        noise_scale = noise_scales[run_idx % len(noise_scales)]
        
        # Time limit per run
        time_per_run = max(5, (time_budget - (time.time() - start_time)) / max(1, n_runs - run_idx))
        
        try:
            bh_energy, bh_pos = _basin_hopping(
                positions, potential, rng,
                max_iterations=30,
                noise_scale=noise_scale,
                accept_threshold=0.05,
                time_limit=time_per_run
            )
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except Exception:
            continue
    
    # If basin hopping didn't find anything, fallback to simple relax of random
    if best_positions is None:
        for seed in range(10):
            positions = _generate_random_cluster(n, seed, min_dist=0.5)
            if positions is not None:
                try:
                    energy, pos = relax(positions, potential)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = pos
                except Exception:
                    continue
    
    # Final fallback
    if best_positions is None:
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-3, 3, (n, 3))
    
    return best_positions
