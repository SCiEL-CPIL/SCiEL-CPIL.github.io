"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def _compute_radius(positions):
    """Compute the maximum radius for given positions.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|))
    """
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(positions)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0
    
    # Pairwise distances using KD-tree for efficiency
    tree = cKDTree(positions)
    # Query for nearest neighbors (excluding self)
    dists, _ = tree.query(positions, k=2)
    # dists[:, 1] is the distance to the nearest other point
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_wall, min_pair)


def _lattice_gas_init(n, rng):
    """Initialize spheres on a lattice gas configuration."""
    # Create a fine grid and randomly assign spheres to cells
    # For 100 spheres in a unit cube, use a grid that allows good spacing
    # Try different grid resolutions
    
    best_init = None
    best_score = -1
    
    for m in [5, 6, 7, 8]:
        # Create grid points
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        if len(grid_points) < n:
            continue
        
        # Randomly select n positions from grid
        idx = rng.choice(len(grid_points), size=n, replace=False)
        positions = grid_points[idx].copy()
        
        score = _compute_radius(positions)
        if score > best_score:
            best_score = score
            best_init = positions
    
    if best_init is None:
        # Fallback: random positions
        best_init = rng.uniform(-0.45, 0.45, size=(n, 3))
    
    return best_init


def _lattice_gas_annealing(positions, n, rng, max_iters=500000, t0=0.02, t1=0.00001):
    """Lattice gas Metropolis-Hastings annealing.
    
    Energy = -radius (we minimize energy = maximize radius)
    Moves:
    1. Single particle displacement
    2. Pair swap (exchange positions of two particles)
    3. Multi-particle displacement
    """
    current = positions.copy()
    current_energy = -_compute_radius(current)
    best = current.copy()
    best_energy = current_energy
    
    accepted = 0
    total = 0
    
    for it in range(max_iters):
        # Geometric cooling
        t = t0 * (t1 / t0) ** (it / max_iters)
        
        # Choose move type
        move_type = rng.random()
        
        proposed = current.copy()
        
        if move_type < 0.6:
            # Single particle displacement
            idx = rng.integers(0, n)
            # Displacement magnitude scales with temperature
            step = 0.05 + 0.1 * (t / t0)
            delta = rng.normal(0, step, size=3)
            proposed[idx] += delta
            proposed[idx] = np.clip(proposed[idx], -0.49, 0.49)
            
        elif move_type < 0.85:
            # Pair swap: exchange positions of two particles
            i, j = rng.choice(n, size=2, replace=False)
            proposed[i], proposed[j] = current[j].copy(), current[i].copy()
            # Add small perturbation to both
            step = 0.02 + 0.05 * (t / t0)
            proposed[i] += rng.normal(0, step, size=3)
            proposed[j] += rng.normal(0, step, size=3)
            proposed[i] = np.clip(proposed[i], -0.49, 0.49)
            proposed[j] = np.clip(proposed[j], -0.49, 0.49)
            
        else:
            # Multi-particle displacement (2-3 particles)
            n_move = rng.integers(2, 4)
            idx = rng.choice(n, size=n_move, replace=False)
            step = 0.03 + 0.08 * (t / t0)
            delta = rng.normal(0, step, size=(n_move, 3))
            proposed[idx] += delta
            proposed[idx] = np.clip(proposed[idx], -0.49, 0.49)
        
        proposed_energy = -_compute_radius(proposed)
        
        total += 1
        delta_e = proposed_energy - current_energy
        
        # Metropolis acceptance
        if delta_e <= 0:
            # Better or equal energy, always accept
            current = proposed
            current_energy = proposed_energy
            accepted += 1
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        else:
            # Worse energy, accept with probability exp(-delta_e / T)
            if delta_e < 100 * t:  # Avoid overflow
                prob = np.exp(-delta_e / max(t, 1e-10))
                if rng.random() < prob:
                    current = proposed
                    current_energy = proposed_energy
                    accepted += 1
    
    return best, -best_energy


def _local_refinement(positions, n, rng, time_budget=30):
    """Local refinement with small perturbations."""
    current = positions.copy()
    current_score = _compute_radius(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Try small perturbations
        idx = rng.integers(0, n)
        proposed = current.copy()
        proposed[idx] += rng.normal(0, 0.005, size=3)
        proposed[idx] = np.clip(proposed[idx], -0.49, 0.49)
        
        proposed_score = _compute_radius(proposed)
        
        if proposed_score > current_score:
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Sometimes accept worse to escape local minima
            if rng.random() < 0.1:
                current = proposed
                current_score = proposed_score
    
    return best, best_score


def construct():
    """Construct positions for 100 equal spheres in a unit cube using lattice gas annealing."""
    n = N
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    time_budget = 110  # Leave 10s buffer for safety
    
    # Generate initial configuration
    positions = _lattice_gas_init(n, rng)
    best_positions = positions.copy()
    best_score = _compute_radius(positions)
    
    # Run lattice gas annealing
    elapsed = time.time() - start_time
    if elapsed < time_budget:
        pos_annealed, score_annealed = _lattice_gas_annealing(
            positions, n, rng,
            max_iters=300000,
            t0=0.015,
            t1=0.000005
        )
        
        if score_annealed > best_score:
            best_positions = pos_annealed
            best_score = score_annealed
    
    # Check time and run another round if possible
    elapsed = time.time() - start_time
    if elapsed < time_budget - 10:
        pos_annealed2, score_annealed2 = _lattice_gas_annealing(
            best_positions, n, rng,
            max_iters=200000,
            t0=0.005,
            t1=0.000001
        )
        
        if score_annealed2 > best_score:
            best_positions = pos_annealed2
            best_score = score_annealed2
    
    # Final local refinement
    elapsed = time.time() - start_time
    if elapsed < time_budget - 5:
        remaining_time = min(30, time_budget - elapsed)
        pos_refined, score_refined = _local_refinement(
            best_positions, n, rng,
            time_budget=remaining_time
        )
        
        if score_refined > best_score:
            best_positions = pos_refined
            best_score = score_refined
    
    return best_positions
# EVOLVE-BLOCK-END
