"""Improved Dzugutov cluster global minima search using parallel basin-hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16

MIN_DIST = 0.5
DEADLINE_MARGIN = 10  # seconds before the 330s kill


def _dzugutov_energy(pos):
    """Vectorized Dzugutov energy computation."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Check minimum distance
    if np.min(dists) < MIN_DIST:
        return 1e10
    
    E = 0.0
    # Region 1: r < a
    mask_a = dists < a_POT
    if np.any(mask_a):
        r_a = dists[mask_a]
        E += np.sum(A_POT * (r_a ** (-m_POT) - B_POT) * np.exp(c_POT / (r_a - a_POT)))
    
    # Region 2: a <= r < b
    mask_b = (dists >= a_POT) & (dists < b_POT)
    if np.any(mask_b):
        r_b = dists[mask_b]
        E += np.sum(B_POT * np.exp(d_POT / (r_b - b_POT)))
    
    return E


def _generate_polytetrahedral_seed(n, rng):
    """Generate a compact cluster with polytetrahedral character."""
    # Start with a small tetrahedral core and add atoms in shells
    # This mimics the polytetrahedral packing favored by Dzugutov
    
    # First, create a dense core
    points = []
    
    # Central atom
    points.append(np.zeros(3))
    
    # Tetrahedral neighbors (4 atoms)
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / np.sqrt(3.0)
    spacing = 1.15  # slightly larger than equilibrium to avoid overlap
    for t in tetra:
        points.append(spacing * t)
    
    # Add more atoms in icosahedral/polytetrahedral fashion
    # Use a Fibonacci sphere for even distribution on shells
    while len(points) < n:
        # Choose shell radius
        shell_idx = len(points) // 6
        shell_r = spacing * (1.0 + 0.5 * shell_idx)
        
        # Generate candidate points on a sphere
        for _ in range(50):
            # Random direction
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            candidate = shell_r * direction
            
            # Check minimum distance
            valid = True
            for p in points:
                if np.linalg.norm(candidate - p) < MIN_DIST + 0.05:
                    valid = False
                    break
            
            if valid:
                points.append(candidate)
                if len(points) >= n:
                    break
    
    return np.array(points[:n])


def _perturb(pos, step_size, rng):
    """Perturb positions with Gaussian noise."""
    perturbed = pos + rng.normal(0, step_size, pos.shape)
    return perturbed


def _basin_hopping(pos0, potential, parameter, rng, max_iters=50, step_size=0.3, 
                   cooling=0.95, temp0=0.5, deadline=None):
    """Basin-hopping algorithm for local minima search."""
    best_pos = pos0.copy()
    best_energy = _dzugutov_energy(pos0)
    
    # Initial relaxation
    energy, relaxed_pos = relax(pos0, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_pos = relaxed_pos.copy()
    
    current_pos = relaxed_pos.copy()
    current_energy = energy
    temp = temp0
    step = step_size
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        perturbed = _perturb(current_pos, step, rng)
        
        # Relax
        try:
            new_energy, new_pos = relax(perturbed, potential, parameter)
        except Exception:
            new_energy = current_energy + 1.0
            new_pos = perturbed
        
        # Metropolis acceptance
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / max(temp, 1e-10)):
            current_pos = new_pos
            current_energy = new_energy
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
        
        # Cool down
        temp *= cooling
        step *= 0.98
    
    return best_energy, best_pos


def _worker(args):
    """Worker function for parallel basin-hopping."""
    n, potential, parameter, seed, deadline = args
    rng = np.random.default_rng(seed)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate multiple starting configurations
    n_starts = 3
    for s in range(n_starts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate seed
        pos0 = _generate_polytetrahedral_seed(n, rng)
        
        # Basin hopping
        energy, pos = _basin_hopping(
            pos0, potential, parameter, rng,
            max_iters=30, step_size=0.25,
            cooling=0.93, temp0=0.4,
            deadline=deadline
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov clusters using parallel basin-hopping."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 330 - DEADLINE_MARGIN
    
    # Use 4 threads for parallel exploration
    n_threads = 4
    
    # Create worker arguments with different seeds
    base_seed = n * 1000
    worker_args = []
    for t in range(n_threads):
        worker_args.append((n, potential, parameter, base_seed + t * 7919, deadline))
    
    best_energy = np.inf
    best_pos = None
    
    # Run parallel basin-hopping
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(_worker, args): i for i, args in enumerate(worker_args)}
        
        for future in as_completed(futures):
            try:
                energy, pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception as e:
                pass
    
    # If we still have time, do additional refinement
    if best_pos is not None and time.time() < deadline - 5:
        # Final relaxation of the best solution
        try:
            final_energy, final_pos = relax(best_pos, potential, parameter)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = final_pos
        except Exception:
            pass
    
    # If no valid result was found, generate a fallback
    if best_pos is None:
        rng = np.random.default_rng(n)
        pos = _generate_polytetrahedral_seed(n, rng)
        try:
            _, pos = relax(pos, potential, parameter)
        except Exception:
            pass
        best_pos = pos
    
    return best_pos
# EVOLVE-BLOCK-END
