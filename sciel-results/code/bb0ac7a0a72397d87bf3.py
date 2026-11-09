"""
Strategy: Parallel Simulated Annealing with Vectorized Dzugutov Potential and Distance Penalty.
The Dzugutov potential is designed to favor PT order (icosahedral shells).
Instead of relying on gradient-based local minimization (L-BFGS-B) which can get trapped in local minima,
we use Simulated Annealing (SA) which is robust to rugged landscapes and can escape local minima via thermal fluctuations.
1. Generate diverse initial structures (PT shells, random compact).
2. Use a parallel pool (up to 4 threads) to run independent SA chains.
3. Use a vectorized, differentiable (via numerical gradient or just energy) energy function with a penalty for < 0.5 Å distances.
4. Track the global best energy and return the corresponding positions.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
import threading

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 290.0  # Safety margin for 300s limit
_lock = threading.Lock()

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """
    Compute the energy of the Dzugutov potential.
    Vectorized for speed.
    """
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist = dist[mask]
    
    if len(dist) == 0:
        return 0.0
    
    # Avoid division by zero
    dist = np.maximum(dist, 1e-8)
    
    energy = 0.0
    
    # Term 1: A(r^-m - B)exp(c/(r - a)) [r < a]
    mask_a = dist < a
    if np.any(mask_a):
        r_a = dist[mask_a]
        v_a = A * (r_a ** (-m) - B) * np.exp(c / (r_a - a))
        energy += np.sum(v_a)
        
    # Term 2: B exp(d/(r - b)) [r < b]
    mask_b = dist < b
    if np.any(mask_b):
        r_b = dist[mask_b]
        v_b = B * np.exp(d / (r_b - b))
        energy += np.sum(v_b)
        
    return energy

def _penalized_energy(positions, penalty_strength=1000.0):
    """
    Energy with a penalty for atoms closer than 0.5 Å.
    """
    base_energy = _dzugutov_energy(positions)
    
    # Penalty for close contacts
    n = len(positions)
    if n < 2:
        return base_energy
        
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist = dist[mask]
    
    # Penalty: sum of (0.5 - r)^2 for r < 0.5
    close_mask = dist < 0.5
    if np.any(close_mask):
        close_dist = dist[close_mask]
        penalty = penalty_strength * np.sum((0.5 - close_dist) ** 2)
    else:
        penalty = 0.0
        
    return base_energy + penalty

def _generate_icosahedron(n_atoms, rng):
    """
    Generate a structure based on icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)]
    
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)
        
    if n_atoms <= 13:
        return np.array(positions[:n_atoms])
        
    # For n > 13, add atoms on a second shell
    r2 = 1.9 * 1.7
    num_outer = n_atoms - 13
    if num_outer > 0:
        outer_pts = []
        for _ in range(num_outer):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            outer_pts.append(v)
        positions.extend(outer_pts)
        
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng, spacing=1.1):
    """
    Generate a random compact structure.
    """
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    max_iters = 10000
    for _ in range(max_iters):
        if len(points) >= n_atoms:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
            points.append(p)
    # If we didn't get enough, just add random points
    while len(points) < n_atoms:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _sa_chain(n_atoms, rng, initial_pos, time_limit):
    """
    Run a single Simulated Annealing chain.
    """
    start_time = time.time()
    current_pos = initial_pos.copy()
    current_energy = _penalized_energy(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Annealing schedule
    T_start = 1.0
    T_end = 0.001
    alpha = 0.999
    
    T = T_start
    step_scale = 0.3
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
            
        # Propose a new position
        new_pos = current_pos + rng.normal(0, step_scale, size=current_pos.shape)
        
        # Compute energies
        new_energy = _penalized_energy(new_pos)
        
        # Metropolis criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.uniform() < np.exp(-delta_E / T):
            current_pos = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
        # Update temperature
        T *= alpha
        step_scale *= 0.9995
        
    return best_pos, best_energy

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    time_budget = _get_remaining_time()
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starts
    starts = []
    
    # 1. PT-inspired
    try:
        pt_pos = _generate_icosahedron(size, rng)
        starts.append(pt_pos)
    except Exception:
        pass
        
    # 2. Random Compact
    try:
        rc_pos = _random_compact(size, rng)
        starts.append(rc_pos)
    except Exception:
        pass
        
    # 3. Perturbed PT
    try:
        pt_pos2 = _generate_icosahedron(size, rng)
        pt_pos2 = pt_pos2 + rng.normal(0, 0.3, size=pt_pos2.shape)
        starts.append(pt_pos2)
    except Exception:
        pass
        
    # 4. Another Random Compact
    try:
        rc_pos2 = _random_compact(size, rng)
        starts.append(rc_pos2)
    except Exception:
        pass
        
    # Use parallel pool
    n_workers = min(4, len(starts))
    time_per_chain = time_budget / (n_workers * 1.5)  # Leave some margin
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            initial_pos = starts[i]
            # Create a new RNG for each thread to avoid contention
            chain_rng = np.random.default_rng(size + i)
            future = executor.submit(_sa_chain, size, chain_rng, initial_pos, time_per_chain)
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            try:
                pos, energy = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
                
    # If no valid result, fallback to the first start
    if best_positions is None:
        best_positions = starts[0]
        
    return best_positions
# EVOLVE-BLOCK-END
