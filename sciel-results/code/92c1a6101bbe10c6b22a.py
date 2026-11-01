"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def _compute_clearances(positions):
    """Compute all pairwise distances and wall clearances.
    Returns (min_pair_dist, min_wall_dist, pair_dists, wall_dists)
    """
    n = len(positions)
    if n == 0:
        return 0.0, 0.0, None, None
    
    # Wall distances: for each sphere, distance to nearest wall
    wall_dists = 0.5 - np.abs(positions)  # shape (n, 3)
    min_wall_per_sphere = np.min(wall_dists, axis=1)  # shape (n,)
    min_wall = np.min(min_wall_per_sphere)
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min_pair, min_wall, dists, wall_dists


def _score_fast(positions):
    """Fast score: only compute what's needed."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(positions)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0
    
    # Pairwise distances - use a more efficient approach
    # For n=100, this is 4950 pairs, manageable
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)


def _find_bottleneck(positions, current_r):
    """Find which constraints are active (limiting the radius).
    Returns (active_pairs, active_walls) where active_pairs is list of (i,j) 
    and active_walls is list of (i, axis) for constraints at distance <= current_r + eps
    """
    n = len(positions)
    eps = 1e-6
    active_pairs = []
    active_walls = []
    
    # Check pairwise
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] / 2.0 <= current_r + eps:
                active_pairs.append((i, j))
    
    # Check walls
    for i in range(n):
        for k in range(3):
            if 0.5 - abs(positions[i, k]) <= current_r + eps:
                active_walls.append((i, k))
    
    return active_pairs, active_walls


def _generate_grid_init(n, m, rng):
    """Generate grid-based initialization with random jitter."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if n <= len(grid):
        idx = rng.choice(len(grid), size=n, replace=False)
        positions = grid[idx].copy()
        # Add small random jitter
        positions += rng.normal(0, 0.01, size=positions.shape)
        positions = np.clip(positions, -0.5, 0.5)
        return positions
    else:
        m2 = int(np.ceil(n ** (1.0/3.0)))
        ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
        grid2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
        if n <= len(grid2):
            idx = rng.choice(len(grid2), size=n, replace=False)
            positions = grid2[idx].copy()
            positions += rng.normal(0, 0.01, size=positions.shape)
            positions = np.clip(positions, -0.5, 0.5)
            return positions
        else:
            return rng.uniform(-0.5, 0.5, size=(n, 3))


def _anneal_with_moves(positions, n, max_iters, t0, t1, step0, rng, time_budget):
    """Improved annealing with smarter moves."""
    current = positions.copy()
    current_score = _score_fast(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    
    for it in range(max_iters):
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        # Geometric cooling
        t = t0 * (t1 / t0) ** (it / max_iters)
        step = step0 * (t / t0) ** 0.3
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single sphere move
            idx = rng.integers(0, n)
            proposed = current.copy()
            delta = rng.normal(0, step, size=3)
            proposed[idx] += delta
            proposed = np.clip(proposed, -0.5, 0.5)
            
        elif move_type < 0.8:
            # Pair swap-like move: move two spheres in opposite directions
            i, j = rng.choice(n, size=2, replace=False)
            proposed = current.copy()
            delta = rng.normal(0, step, size=3)
            proposed[i] += delta
            proposed[j] -= delta * 0.5  # Partial opposite
            proposed = np.clip(proposed, -0.5, 0.5)
            
        else:
            # Local nudge: try moving a sphere away from its nearest neighbor
            idx = rng.integers(0, n)
            # Find nearest neighbor
            dists_to_others = np.sqrt(np.sum((current - current[idx]) ** 2, axis=1))
            dists_to_others[idx] = np.inf
            nearest = np.argmin(dists_to_others)
            
            # Direction away from nearest
            direction = current[idx] - current[nearest]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
            
            proposed = current.copy()
            proposed[idx] += direction * step * 0.5
            proposed = np.clip(proposed, -0.5, 0.5)
        
        proposed_score = _score_fast(proposed)
        
        # Metropolis acceptance
        if proposed_score >= current_score:
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta_e = proposed_score - current_score
            if delta_e < 0:
                accept_prob = np.exp(delta_e / max(t, 1e-10))
                if rng.random() < accept_prob:
                    current = proposed
                    current_score = proposed_score
    
    return best, best_score


def _local_refinement(positions, n, max_iters, rng, time_budget):
    """Fine-grained local search."""
    current = positions.copy()
    current_score = _score_fast(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Try small moves on each sphere
        idx = rng.integers(0, n)
        step = 0.005 + 0.005 * rng.random()
        
        # Try 8 directions (±x, ±y, ±z) plus random
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
            rng.normal(size=3)
        ]
        
        best_move = None
        best_move_score = current_score
        
        for d in directions:
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                continue
            d = d / norm
            proposed = current.copy()
            proposed[idx] += d * step
            proposed = np.clip(proposed, -0.5, 0.5)
            s = _score_fast(proposed)
            if s > best_move_score:
                best_move_score = s
                best_move = proposed.copy()
        
        if best_move is not None:
            current = best_move
            current_score = best_move_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
    
    return best, best_score


def construct():
    """Construct positions for 100 equal spheres in a unit cube."""
    n = N
    start_time = time.time()
    time_budget = 110  # Leave 10s buffer
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initializations
    best_positions = None
    best_score = -np.inf
    
    # Try different grid sizes
    for m in [4, 5, 6]:
        for trial in range(2):
            positions = _generate_grid_init(n, m, rng)
            score = _score_fast(positions)
            if score > best_score:
                best_score = score
                best_positions = positions.copy()
    
    # Also try random initialization
    positions_rand = rng.uniform(-0.45, 0.45, size=(n, 3))
    score_rand = _score_fast(positions_rand)
    if score_rand > best_score:
        best_score = score_rand
        best_positions = positions_rand.copy()
    
    # Phase 1: Coarse annealing
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 30:
        pos1, score1 = _anneal_with_moves(
            best_positions, n,
            max_iters=200000,
            t0=0.08, t1=0.0001,
            step0=0.15,
            rng=rng,
            time_budget=min(remaining - 10, 50)
        )
        if score1 > best_score:
            best_score = score1
            best_positions = pos1
    
    # Phase 2: Medium annealing
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 20:
        pos2, score2 = _anneal_with_moves(
            best_positions, n,
            max_iters=150000,
            t0=0.03, t1=0.00001,
            step0=0.06,
            rng=rng,
            time_budget=min(remaining - 10, 40)
        )
        if score2 > best_score:
            best_score = score2
            best_positions = pos2
    
    # Phase 3: Fine annealing
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 10:
        pos3, score3 = _anneal_with_moves(
            best_positions, n,
            max_iters=100000,
            t0=0.01, t1=0.000001,
            step0=0.02,
            rng=rng,
            time_budget=min(remaining - 5, 20)
        )
        if score3 > best_score:
            best_score = score3
            best_positions = pos3
    
    # Phase 4: Local refinement
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 5:
        pos4, score4 = _local_refinement(
            best_positions, n,
            max_iters=50000,
            rng=rng,
            time_budget=min(remaining - 2, 10)
        )
        if score4 > best_score:
            best_score = score4
            best_positions = pos4
    
    # Final check
    final_score = _score_fast(best_positions)
    if final_score > best_score:
        best_score = final_score
    
    return best_positions
# EVOLVE-BLOCK-END
