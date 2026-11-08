# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_random_compact(n, rng, min_dist=0.65):
    """Generate a random compact cluster with minimum distance constraints."""
    points = []
    radius = 1.2 * n ** (1.0 / 3.0) + 1.0
    max_attempts = 5000 * n
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        if not points:
            points.append(p)
        else:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) >= min_dist:
                points.append(p)
            else:
                attempts += 1
    if len(points) < n:
        # Fallback: simple grid
        points = []
        side = int(np.ceil(n ** (1/3))) + 1
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if len(points) < n:
                        points.append(np.array([i, j, k]) * 0.8)
        points = np.array(points[:n])
        points -= points.mean(axis=0)
    else:
        points = np.array(points)
        points -= points.mean(axis=0)
    return points


def _generate_grid_cluster(n, rng):
    """Generate a grid-like cluster, faster than random compact for large n."""
    points = []
    side = int(np.ceil(n ** (1/3))) + 2
    spacing = 0.9
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(points) < n:
                    points.append(np.array([i, j, k]) * spacing)
    if len(points) < n:
        # Fill remaining with random
        while len(points) < n:
            p = rng.uniform(-side*spacing, side*spacing, 3)
            points.append(p)
    points = np.array(points[:n])
    # Add small noise to break symmetry
    points += rng.normal(0, 0.1, points.shape)
    points -= points.mean(axis=0)
    return points


def _basin_hopping_worker(initial_pos, kind, rng_seed, time_limit, max_iter=50):
    """
    Perform basin hopping from initial_pos within time_limit.
    """
    start_time = time.time()
    current_energy, current_pos = relax(initial_pos, 'sutton_chen', kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    rng = np.random.default_rng(rng_seed)
    n = current_pos.shape[0]
    
    # Adaptive step size
    step_size = 0.4
    k_atoms = max(2, int(0.1 * n))
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb
        perturbed = current_pos.copy()
        if k_atoms < n:
            indices = rng.choice(n, size=k_atoms, replace=False)
            perturbed[indices] += rng.normal(0, step_size, (k_atoms, 3))
        else:
            perturbed += rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            step_size *= 0.9
        else:
            delta_e = new_energy - current_energy
            temp = 0.5 * np.exp(-i / 10.0) + 0.05
            if rng.random() < np.exp(-delta_e / temp):
                current_energy = new_energy
                current_pos = new_pos
                step_size *= 1.1
            else:
                step_size *= 0.9
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Safety check for min distance
        if n > 1:
            min_dist = pdist(current_pos).min()
            if min_dist < 0.35:
                # Nudge apart
                for _ in range(3):
                    current_pos += rng.normal(0, 0.1, current_pos.shape)
                    min_dist = pdist(current_pos).min()
                    if min_dist >= 0.35:
                        break
                current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Parallel multi-start basin-hopping search for Sutton-Chen cluster global minima.
    """
    start_time = time.time()
    # Safety margin: Return by 300s to be safe against the 330s kill
    deadline = start_time + 280  

    n = instance["n"]
    kind = instance["kind"]
    
    rng_master = np.random.default_rng(42)
    
    # Generate a few diverse initial structures quickly
    initial_structures = []
    
    # 1. Random compact
    initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.7))
    
    # 2. Grid cluster
    initial_structures.append(_generate_grid_cluster(n, rng_master))
    
    # 3. Another random compact with different seed
    rng2 = np.random.default_rng(123)
    initial_structures.append(_generate_random_compact(n, rng2, min_dist=0.6))
    
    # 4. Another grid with noise
    rng3 = np.random.default_rng(456)
    initial_structures.append(_generate_grid_cluster(n, rng3))
    
    results = []
    
    # Use 4 workers, but strictly time-limited
    # Total budget 280s. 4 workers. 10s setup overhead? No, setup is fast.
    # Let's give each worker ~60s. 4 * 60 = 240s. Leaves 40s buffer.
    worker_time_limit = 60.0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(4):
            start_pos = initial_structures[i % len(initial_structures)]
            seed = 1000 + i
            futures.append(executor.submit(_basin_hopping_worker, start_pos, kind, seed, worker_time_limit, max_iter=100))
        
        for future in futures:
            try:
                # Wait for result, but check global time
                remaining = deadline - time.time()
                if remaining < 5:
                    break
                energy, pos = future.result(timeout=max(1, remaining))
                if np.isfinite(energy):
                    results.append((energy, pos))
            except Exception:
                pass
    
    if not results:
        # Fallback if all workers failed
        return _generate_random_compact(n, rng_master)
    
    # Find the best result
    best_energy, best_pos = min(results, key=lambda x: x[0])
    
    # Final polish if time remains
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            # Quick local polish
            for _ in range(3):
                if time.time() - start_time > deadline - 5:
                    break
                rng_polish = np.random.default_rng(999)
                perturbed = best_pos + rng_polish.normal(0, 0.05, best_pos.shape)
                new_e, new_p = relax(perturbed, 'sutton_chen', kind)
                if new_e < best_energy:
                    best_energy = new_e
                    best_pos = new_p
        except Exception:
            pass
    
    # Final validity check
    if n > 1:
        min_dist = pdist(best_pos).min()
        if min_dist < 0.35:
            rng_fix = np.random.default_rng(123)
            for _ in range(10):
                best_pos += rng_fix.normal(0, 0.05, best_pos.shape)
                min_dist = pdist(best_pos).min()
                if min_dist >= 0.35:
                    break
            try:
                best_energy, best_pos = relax(best_pos, 'sutton_chen', kind)
            except Exception:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
