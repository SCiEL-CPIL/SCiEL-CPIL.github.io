"""Dzugutov cluster global minima search using parallel Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
import math
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for a cluster. Vectorized for speed."""
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    # Get upper triangle indices
    i, j = np.triu_indices(n, k=1)
    r_ij = dists[i, j]
    
    energy = 0.0
    # Vectorized potential calculation
    # Mask for r < a
    mask_a = r_ij < a_POT
    if np.any(mask_a):
        r_a = r_ij[mask_a]
        energy += np.sum(A_POT * (r_a ** (-m_POT) - B_POT) * np.exp(c_POT / (r_a - a_POT)))
        
    # Mask for a <= r < b
    mask_b = (r_ij >= a_POT) & (r_ij < b_POT)
    if np.any(mask_b):
        r_b = r_ij[mask_b]
        energy += np.sum(B_POT * np.exp(d_POT / (r_b - b_POT)))
        
    return float(energy)


def _generate_seed(n_atoms, variant, seed_idx):
    """Generate a diverse initial structure."""
    rng = np.random.default_rng(seed_idx)
    
    if variant == 0:
        # Random compact packing
        radius = 1.2 * (n_atoms / 4.18879) ** (1/3) # Rough volume estimate
        points = []
        attempts = 0
        while len(points) < n_atoms and attempts < 10000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.8 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) < n_atoms:
            # Fallback: force fit
            while len(points) < n_atoms:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
        return np.array(points)
        
    elif variant == 1:
        # Icosahedral-based
        phi = (1 + np.sqrt(5)) / 2
        ico_vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0]) * 1.1
        points = list(ico_vertices)
        # Add more atoms randomly around
        while len(points) < n_atoms:
            base = rng.choice(len(ico_vertices))
            p = ico_vertices[base] + rng.normal(0, 0.5, 3)
            if all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
            else:
                # If collision, just add and let relaxation fix it
                points.append(p)
        return np.array(points[:n_atoms])
        
    elif variant == 2:
        # BCC-like lattice
        points = []
        spacing = 1.1
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    if len(points) >= n_atoms:
                        break
                    p = np.array([i, j, k]) * spacing
                    if (i+j+k) % 2 == 0: # BCC condition
                        points.append(p)
        # Fill remaining with random
        while len(points) < n_atoms:
            points.append(rng.normal(0, 2, 3))
        return np.array(points[:n_atoms])
        
    else:
        # FCC-like lattice
        points = []
        spacing = 1.1
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    if len(points) >= n_atoms:
                        break
                    p = np.array([i, j, k]) * spacing
                    if (i+j+k) % 2 == 0 or (i+j+k) % 2 == 1: # Just dense
                        points.append(p)
        while len(points) < n_atoms:
            points.append(rng.normal(0, 2, 3))
        return np.array(points[:n_atoms])


def _simulate_annealing(n_atoms, seed_positions, time_limit, seed_idx):
    """Run a single Simulated Annealing chain."""
    rng = np.random.default_rng(seed_idx)
    start_time = time.time()
    
    # Initial relaxation
    try:
        current_energy, current_positions = relax(seed_positions, 'dzugutov')
    except Exception:
        return np.inf, seed_positions

    best_energy = current_energy
    best_positions = current_positions.copy()
    
    # SA Parameters
    T_init = 2.0
    T_final = 0.01
    alpha = 0.98 # Cooling rate
    T = T_init
    step_size = 0.2
    
    while time.time() - start_time < time_limit:
        if T < T_final:
            break
            
        # Generate perturbation
        # Adaptive step size: larger when T is high
        current_step = step_size * (T / T_init + 0.2)
        new_positions = current_positions + rng.normal(0, current_step, current_positions.shape)
        
        # Ensure minimum distance 0.5
        # Simple pairwise check is O(N^2) but N<=60 so it's fine
        # We can vectorize this check
        diff = new_positions[:, np.newaxis, :] - new_positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        i, j = np.where(np.triu(dists < 0.5, k=1))
        for idx in range(len(i)):
            ii, jj = i[idx], j[idx]
            d = dists[ii, jj]
            if d < 1e-10:
                direction = rng.normal(0, 1, 3)
            else:
                direction = (new_positions[ii] - new_positions[jj]) / d
            push = (0.5 - d) / 2
            new_positions[ii] += push * direction
            new_positions[jj] -= push * direction
            
        # Relax
        try:
            new_energy, relaxed_positions = relax(new_positions, 'dzugutov')
        except Exception:
            new_energy = np.inf
            relaxed_positions = new_positions

        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < math.exp(-delta_e / T)
            
        if accept:
            current_energy = new_energy
            current_positions = relaxed_positions
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
                
        T *= alpha
        # Occasionally increase step size if stuck? 
        # For now, keep it simple.
        
    return best_energy, best_positions


def construct(instance):
    """Find low-energy Dzugutov cluster structures using parallel Simulated Annealing."""
    size = instance["n"]
    start_time = time.time()
    total_time_budget = 45.0 # Leave margin for 6 instances * 45s = 270s < 300s
    
    # Define number of parallel chains
    n_chains = 4
    
    # Generate diverse seeds
    seeds = []
    for i in range(n_chains):
        variant = i % 4
        seeds.append(_generate_seed(size, variant, seed_idx=size * 100 + i))
        
    # Relax seeds initially to get better starting points
    relaxed_seeds = []
    for s in seeds:
        try:
            _, rel = relax(s, 'dzugutov')
            relaxed_seeds.append(rel)
        except Exception:
            relaxed_seeds.append(s)
            
    # Run parallel SA
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_chains) as executor:
        futures = []
        time_per_chain = total_time_budget / n_chains * 0.9 # 10% overhead buffer
        for i in range(n_chains):
            future = executor.submit(_simulate_annealing, size, relaxed_seeds[i], time_per_chain, size * 1000 + i)
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            try:
                energy, positions = future.result()
                results.append((energy, positions))
            except Exception:
                pass
                
    if not results:
        # Fallback: return relaxed seed
        return relaxed_seeds[0]
        
    # Find best result
    best_result = min(results, key=lambda x: x[0])
    best_positions = best_result[1]
    
    # Final relaxation
    try:
        _, final_positions = relax(best_positions, 'dzugutov')
        return final_positions
    except Exception:
        return best_positions
# EVOLVE-BLOCK-END
