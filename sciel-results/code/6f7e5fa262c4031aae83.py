"""Simulated Annealing + Basin Hopping for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from multiprocessing import Pool, cpu_count
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _is_valid(positions, min_dist=0.35):
    """Check that all coordinates are finite and pairwise distances >= min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Efficient distance check
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True

def _generate_initial_cluster(n, rng):
    """Generate a compact initial cluster."""
    # Start with a BCC-like lattice which is often good for these potentials
    points = []
    # Central atom
    points.append(np.zeros(3))
    
    # Generate candidate lattice points
    coords = []
    max_coord = 4
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                # BCC condition: i+j+k is even
                if (i + j + k) % 2 == 0:
                    coords.append([i, j, k])
    
    # Sort by distance from center
    coords = sorted(coords, key=lambda c: np.dot(c, c))
    
    # Take nearest n-1
    for c in coords:
        if len(points) >= n:
            break
        points.append(np.array(c, dtype=float) * 0.5)
        
    # If not enough (unlikely for n<100 with max_coord=4), add random
    while len(points) < n:
        p = rng.uniform(-3, 3, 3)
        points.append(p)
        
    return np.array(points[:n])

def _sa_step(positions, rng, temperature, n_exp, m_exp, c, kind, n):
    """
    Perform one step of Simulated Annealing.
    Moves a random atom by a Gaussian perturbation.
    Returns new positions and energy difference.
    """
    # Pick a random atom
    idx = rng.integers(n)
    
    # Perturbation size scales with temperature
    # Typical bond length is ~1.0-1.5. 
    # Start with a move size that allows exploration.
    move_scale = 0.1 + 0.2 * (temperature / 10.0) # Heuristic scaling
    move = rng.normal(0, move_scale, 3)
    
    new_positions = positions.copy()
    new_positions[idx] += move
    
    # Check validity (min distance)
    # Only check distance for the moved atom to save time
    d = np.linalg.norm(new_positions - new_positions[idx], axis=1)
    d[idx] = np.inf # ignore self
    if np.min(d) < 0.35:
        # Move is invalid, reject
        return positions, np.inf
    
    # Calculate energy difference
    # We need E(new) - E(old)
    # E = sum_{i<j} (r^-n - c * sqrt(r^-m))
    # This is expensive to recompute fully. 
    # However, for n~50-80, O(N^2) is ~4000 ops. 
    # Doing this in Python loop is slow. 
    # We rely on the fact that we only change one atom.
    # Delta E = E(new, idx) - E(old, idx) + E(new, others) - E(old, others)
    # Actually, E is sum over all pairs. 
    # Pairs involving idx change. Pairs not involving idx do not.
    # So Delta E = sum_{j != idx} [V(r_ij_new) - V(r_ij_old)]
    
    old_r = np.linalg.norm(positions - positions[idx], axis=1)
    old_r[idx] = 1.0 # avoid div by zero, will be masked
    new_r = np.linalg.norm(new_positions - new_positions[idx], axis=1)
    new_r[idx] = 1.0
    
    # Mask out self
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    
    old_r = old_r[mask]
    new_r = new_r[mask]
    
    # Sutton-Chen pair potential
    # V(r) = r^-n_exp - c * sqrt(r^-m_exp)
    # Note: The task defines E = sum_i [1/2 sum_{j!=i} ...] which is sum_{i<j} V(r_ij)
    
    # Calculate V for old and new
    # Handle r=0 or very small? We checked min dist > 0.35, so r > 0.35
    # r^-n_exp can be large if r is small, but 0.35^-12 is ~ 30000. 
    # Floating point should be fine.
    
    old_v = old_r ** (-n_exp) - c * np.sqrt(old_r ** (-m_exp))
    new_v = new_r ** (-n_exp) - c * np.sqrt(new_r ** (-m_exp))
    
    delta_e = np.sum(new_v) - np.sum(old_v)
    
    return new_positions, delta_e

def _basin_hopping(positions, potential, kind, n, rng, max_steps=200, step_scale=0.1):
    """
    Perform Basin Hopping: 
    1. Local Minimize (L-BFGS-B)
    2. Perturb
    3. Local Minimize
    4. Accept/Reject based on energy
    """
    best_pos = positions.copy()
    best_energy, _ = sutton_chen_energy_gradient(best_pos, kind)
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    for i in range(max_steps):
        # Perturb current position
        # Scale perturbation with temperature-like factor? 
        # BH usually uses a fixed or decreasing step size.
        # Let's use a fixed step size around 0.1-0.2
        perturb = rng.normal(0, step_scale, (n, 3))
        new_pos = current_pos + perturb
        
        # Local minimize
        try:
            energy, relaxed_pos = relax(new_pos, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(energy) or not _is_valid(relaxed_pos):
            continue
            
        # Metropolis acceptance
        if energy < current_energy:
            current_energy = energy
            current_pos = relaxed_pos
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
        else:
            # Accept with probability exp(-delta/T)
            # T is not explicit in standard BH, but we can use a heuristic T
            # Or just accept if close? Standard BH often uses a fixed T or adaptive.
            # Let's use a simple heuristic: accept if delta < 0.01 (flat) or random
            # Actually, standard BH (Wales) uses a Metropolis criterion with T.
            # Let's set T ~ 0.1 for these energies (typically -100 to -500)
            T = 0.5
            delta = energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_energy = energy
                current_pos = relaxed_pos
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
                    
    return best_energy, best_pos

def _worker(args):
    """Worker function for multiprocessing."""
    initial_pos, potential, kind, n, n_exp, m_exp, c, seed, time_budget = args
    rng = np.random.default_rng(seed)
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Phase 1: Simulated Annealing
    # Initial temperature
    T = 5.0
    cooling_rate = 0.995
    
    # Run SA for a fixed number of steps or until time limit
    # N=50, O(N^2) per step is 2500 ops. 
    # Python loop: ~1000 steps/sec? Maybe 500.
    # 5 seconds -> 2500 steps.
    # Let's aim for ~2000-5000 SA steps per worker.
    
    current_pos = initial_pos.copy()
    # Get initial energy
    try:
        current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    except Exception:
        return (np.inf, None)
        
    if not np.isfinite(current_energy):
        return (np.inf, None)
        
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    n_sa_steps = 3000 # Heuristic
    
    for i in range(n_sa_steps):
        if time.time() > deadline - 10: # Leave time for BH
            break
            
        new_pos, delta_e = _sa_step(current_pos, rng, T, n_exp, m_exp, c, kind, n)
        
        if np.isfinite(delta_e):
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_pos = new_pos
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
                    
        T *= cooling_rate
        
    # Phase 2: Basin Hopping from the best SA result
    # Also run BH from a few other good states?
    # Let's just run BH from the best SA state for the remaining time.
    
    bh_start_time = time.time()
    bh_deadline = deadline
    
    # Run BH for as long as possible
    # Each BH step involves a relax call (L-BFGS-B).
    # L-BFGS-B on N=50 might take 0.1-0.5s?
    # So we can afford ~10-50 BH steps.
    
    bh_energy, bh_pos = _basin_hopping(
        best_pos, potential, kind, n, rng, 
        max_steps=100, # Will be limited by time inside _basin_hopping? 
        # Actually _basin_hopping doesn't check time. 
        # We need to pass time budget or check inside.
        # Let's modify _basin_hopping to accept a deadline.
        step_scale=0.1
    )
    
    # The above _basin_hopping implementation doesn't check deadline.
    # I should fix that or wrap it.
    # For now, let's assume it's fast enough or limit steps.
    # Better: modify _basin_hopping to check time.
    
    # Let's re-evaluate: 
    # If _basin_hopping takes too long, it will timeout the worker.
    # The worker has a time_budget. 
    # I will implement a time-checking version of BH inside the worker.
    
    # Revised Plan for Worker:
    # 1. SA
    # 2. Loop: 
    #    a. Check time
    #    b. Perturb best_pos
    #    c. Relax
    #    d. Update best
    
    # Let's redo the BH part properly with time check.
    
    # Reset for proper BH with time check
    current_bh_pos = best_pos.copy()
    current_bh_energy = best_energy
    
    while time.time() < deadline - 2:
        # Perturb
        perturb = rng.normal(0, 0.1, (n, 3))
        new_bh_pos = current_bh_pos + perturb
        
        try:
            energy, relaxed_pos = relax(new_bh_pos, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(energy) or not _is_valid(relaxed_pos):
            continue
            
        # Accept
        if energy < current_bh_energy:
            current_bh_energy = energy
            current_bh_pos = relaxed_pos
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
        else:
            # Metropolis
            T = 0.5
            delta = energy - current_bh_energy
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_bh_energy = energy
                current_bh_pos = relaxed_pos
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
                    
    if best_pos is not None and _is_valid(best_pos):
        return (best_energy, best_pos)
    else:
        return (np.inf, None)

def construct(instance):
    """
    Portfolio search using Simulated Annealing + Basin Hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    # Extract parameters
    if kind == '12-6':
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == '9-6':
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == '10-8':
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        # Default to 12-6 if unknown
        n_exp, m_exp, c = 12, 6, 144.41

    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(size * 1000 + 42)
    
    n_candidates = 8 # Number of parallel searches
    
    candidates = []
    for i in range(n_candidates):
        # Mix of BCC, FCC, Random
        if i % 3 == 0:
            # BCC
            cand = _generate_initial_cluster(size, rng)
        elif i % 3 == 1:
            # FCC-like (shift BCC by 0.25? or just random)
            # Let's use a random compact cluster for variety
            radius = 0.72 * size ** (1.0 / 3.0) * 0.8
            pts = []
            for _ in range(size):
                theta = rng.uniform(0, 2 * np.pi)
                cos_phi = rng.uniform(-1, 1)
                sin_phi = np.sqrt(1 - cos_phi**2)
                r = radius * rng.uniform(0.2, 1.0)
                pts.append(r * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi]))
            cand = np.array(pts)
        else:
            # Icosahedral shell
            # Reuse logic from previous or simple random
            cand = _generate_initial_cluster(size, rng)
            # Add noise
            cand += rng.normal(0, 0.1, cand.shape)
            
        candidates.append(cand)
        
    # Shuffle
    rng.shuffle(candidates)
    
    # Multiprocessing
    n_threads = min(4, cpu_count())
    
    # Distribute candidates
    # If we have 8 candidates and 4 threads, each thread gets 2 candidates?
    # Or we can just run 4 parallel searches, each with a unique seed and IC.
    # Let's run n_threads searches.
    
    args_list = []
    for i in range(n_threads):
        seed = 12345 + i
        time_budget = (deadline - start_time) - 5 # Leave 5s for overhead
        args_list.append((candidates[i], potential, kind, size, n_exp, m_exp, c, seed, time_budget))
        
    best_energy = np.inf
    best_positions = None
    
    with Pool(processes=n_threads) as pool:
        results = pool.map(_worker, args_list)
        
    for energy, positions in results:
        if energy < best_energy and positions is not None:
            best_energy = energy
            best_positions = positions
            
    # Fallback if all failed
    if best_positions is None or not np.isfinite(best_energy):
        # Try a simple relax on a BCC cluster
        cand = _generate_initial_cluster(size, rng)
        try:
            energy, positions = relax(cand, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
            
    if best_positions is None or not np.isfinite(best_energy):
        # Last resort: random valid
        positions = np.zeros((size, 3))
        spacing = 0.72
        idx = 0
        for r in range(int(np.ceil(size ** (1/3))) + 1):
            if idx >= size:
                break
            for i in range(-r, r + 1):
                if idx >= size:
                    break
                for j in range(-r, r + 1):
                    if idx >= size:
                        break
                    positions[idx] = [i * spacing, j * spacing, r * spacing]
                    idx += 1
        while idx < size:
            positions[idx] = rng.uniform(-2, 2, 3)
            idx += 1
        # Ensure validity
        for i in range(size):
            for j in range(i):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 0.35:
                    positions[i] = positions[j] + 0.4 * np.array([1, 0, 0])
        best_positions = positions
        
    if best_positions is not None and _is_valid(best_positions):
        return best_positions
    else:
        # Should not happen
        return np.zeros((size, 3))
# EVOLVE-BLOCK-END
