"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(positions):
    """Compute the radius for a given set of positions.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(positions)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)


def _compute_clearance(positions):
    """Compute minimum clearance (radius) for a configuration."""
    return _score(positions)


def _lattice_init(n, grid_size=5):
    """Initialize spheres on a lattice with random perturbations."""
    rng = np.random.default_rng(42)
    
    # Create a grid of candidate positions
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if n <= len(grid_points):
        # Select n positions from the grid
        selected_idx = rng.choice(len(grid_points), size=n, replace=False)
        positions = grid_points[selected_idx].copy()
    else:
        # If we need more positions than grid points, use smaller grid
        while n > len(grid_points):
            grid_size += 1
            ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
            grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        selected_idx = rng.choice(len(grid_points), size=n, replace=False)
        positions = grid_points[selected_idx].copy()
    
    # Add small random perturbations
    positions += rng.normal(0, 0.01, size=(n, 3))
    positions = np.clip(positions, -0.5, 0.5)
    
    return positions


def _metropolis_annealing_lattice(positions, n, max_iters=500000, t0=0.02, t1=0.00001, 
                                   step_scale=0.05, time_budget=100):
    """
    Metropolis-Hastings annealing with lattice gas dynamics.
    Energy = -clearance (we want to maximize clearance, so minimize -clearance)
    """
    current = positions.copy()
    current_energy = -_compute_clearance(current)  # Negative clearance as energy
    best = current.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(123)
    start_time = time.time()
    
    # Pre-compute some useful values
    n_move_options = [1, 2, 3]
    
    for it in range(max_iters):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Exponential cooling
        t = t0 * (t1 / t0) ** (it / max_iters)
        
        # Step size scales with temperature
        step = step_scale * (t / t0) ** 0.3
        
        # Choose move type: perturb 1-3 random spheres
        n_move = n_move_options[rng.integers(0, len(n_move_options))]
        idx = rng.choice(n, size=n_move, replace=False)
        
        # Generate proposed positions
        proposed = current.copy()
        delta = rng.normal(0, step, size=(n_move, 3))
        proposed[idx] += delta
        
        # Clamp to valid region
        proposed = np.clip(proposed, -0.49, 0.49)
        
        # Compute proposed energy
        proposed_energy = -_compute_clearance(proposed)
        
        # Metropolis acceptance criterion
        delta_e = proposed_energy - current_energy
        
        if delta_e <= 0:
            # Always accept improvements
            current = proposed
            current_energy = proposed_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        else:
            # Accept with probability exp(-delta_e / T)
            if t > 1e-10:
                accept_prob = np.exp(-delta_e / t)
            else:
                accept_prob = 0.0
            
            if rng.random() < accept_prob:
                current = proposed
                current_energy = proposed_energy
    
    return best, -best_energy


def _local_refinement(positions, n, max_iters=10000, step=0.005, time_budget=30):
    """Local refinement with small steps to fine-tune positions."""
    current = positions.copy()
    current_score = _compute_clearance(current)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.default_rng(456)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Perturb a random sphere
        idx = rng.integers(0, n)
        proposed = current.copy()
        proposed[idx] += rng.normal(0, step, size=3)
        proposed = np.clip(proposed, -0.49, 0.49)
        
        proposed_score = _compute_clearance(proposed)
        
        if proposed_score > current_score:
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        elif rng.random() < np.exp((proposed_score - current_score) / 0.001):
            current = proposed
            current_score = proposed_score
    
    return best, best_score


def construct():
    """Construct positions for 100 equal spheres in a unit cube."""
    n = N
    
    # Start with lattice initialization
    positions = _lattice_init(n, grid_size=5)
    best_positions = positions.copy()
    best_score = _compute_clearance(best_positions)
    
    start_time = time.time()
    total_time_budget = 110  # Leave 10s buffer
    
    # Phase 1: Coarse annealing
    elapsed = time.time() - start_time
    if elapsed < total_time_budget:
        pos1, score1 = _metropolis_annealing_lattice(
            positions, n,
            max_iters=200000,
            t0=0.02,
            t1=0.0001,
            step_scale=0.05,
            time_budget=total_time_budget - (time.time() - start_time)
        )
        if score1 > best_score:
            best_positions = pos1
            best_score = score1
    
    # Phase 2: Finer annealing
    elapsed = time.time() - start_time
    if elapsed < total_time_budget - 10:
        pos2, score2 = _metropolis_annealing_lattice(
            best_positions, n,
            max_iters=150000,
            t0=0.005,
            t1=0.00001,
            step_scale=0.02,
            time_budget=total_time_budget - (time.time() - start_time) - 10
        )
        if score2 > best_score:
            best_positions = pos2
            best_score = score2
    
    # Phase 3: Local refinement
    elapsed = time.time() - start_time
    if elapsed < total_time_budget - 5:
        pos3, score3 = _local_refinement(
            best_positions, n,
            max_iters=20000,
            step=0.003,
            time_budget=total_time_budget - (time.time() - start_time) - 5
        )
        if score3 > best_score:
            best_positions = pos3
            best_score = score3
    
    # Final validation: ensure all positions are within bounds
    best_positions = np.clip(best_positions, -0.5, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
