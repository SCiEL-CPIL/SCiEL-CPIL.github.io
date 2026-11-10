import numpy as np
import time

def _run_aco_vec(prizes, distance_matrix, max_len, eta, n_ants=20, n_iters=30, seed=None):
    """
    Vectorized ACO simulation.
    Returns the max total prize collected by any ant.
    """
    n = len(prizes)
    # Pheromone starts at 1
    tau = np.ones((n, n))
    
    # Ant states
    positions = np.zeros(n_ants, dtype=int) # All start at 0
    visited = np.zeros((n_ants, n), dtype=bool)
    visited[:, 0] = True # Start node is visited
    current_len = np.zeros(n_ants)
    prize_sum = np.zeros(n_ants)
    
    # Precompute return costs
    dist_to_origin = distance_matrix[:, 0]
    
    best_total_prize = 0.0
    
    for _ in range(n_iters):
        # 1. Determine valid moves for all ants
        # Get distances from current positions to all nodes
        # positions shape (n_ants,)
        # distance_matrix[positions, :] shape (n_ants, n)
        dist_from_pos = distance_matrix[positions, :]
        
        # Calculate total path length if ant moves to j
        # current_len shape (n_ants, 1)
        # dist_from_pos shape (n_ants, n)
        # dist_to_origin shape (1, n)
        total_path_len = current_len[:, None] + dist_from_pos + dist_to_origin[None, :]
        
        # Valid if: not visited AND total_path_len <= max_len
        valid = (~visited) & (total_path_len <= max_len)
        
        # Calculate probabilities
        # weight = tau * eta
        weights = tau * eta
        weights = np.where(valid, weights, 0.0)
        
        # Normalize
        sum_weights = weights.sum(axis=1, keepdims=True)
        
        # Handle case where no valid moves (sum_weights == 0)
        # In this case, the ant stops. We can set prob to 0 or handle separately.
        # If sum_weights is 0, the ant stays put (or terminates).
        # The problem says "until no node qualifies". So they stop.
        
        next_positions = positions.copy()
        any_moved = np.zeros(n_ants, dtype=bool)
        
        for a in range(n_ants):
            if sum_weights[a, 0] > 1e-12:
                probs = weights[a] / sum_weights[a, 0]
                # Sample next node
                next_node = np.random.choice(n, p=probs)
                next_positions[a] = next_node
                any_moved[a] = True
            else:
                # Ant stops, stays at current position
                next_positions[a] = positions[a]
                any_moved[a] = False

        # Update states for those that moved
        # We need to update visited, current_len, prize_sum
        # Only for ants that moved
        if np.any(any_moved):
            idx = np.where(any_moved)[0]
            # Update visited
            visited[idx, next_positions[idx]] = True
            # Update current_len: add distance from old pos to new pos
            old_pos = positions[idx]
            new_pos = next_positions[idx]
            current_len[idx] += distance_matrix[old_pos, new_pos]
            # Update prize_sum
            prize_sum[idx] += prizes[new_pos]
            
            # Update positions
            positions = next_positions
        
        # Track best prize in this iteration (for pheromone update)
        best_this_iter = np.max(prize_sum)
        if best_this_iter > best_total_prize:
            best_total_prize = best_this_iter
            
        # Update Pheromone
        # tau = 0.9 * tau + (best prize of that iteration) / 10
        tau = 0.9 * tau + (best_this_iter / 10.0)

    return best_total_prize

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    start_time = time.time()
    deadline = start_time + 310.0 # 20s safety margin
    
    # Preserve random state
    rng_state = np.random.get_state()
    
    # Initial heuristic: Prize / Distance
    # Add small epsilon to avoid division by zero
    dist_safe = distance_matrix + 1e-9
    base_eta = (prizes[None, :] + 1e-9) / dist_safe
    # Ensure positive
    base_eta = np.maximum(base_eta, 1e-9)
    
    current_eta = base_eta.copy()
    
    # Evaluate base
    base_score = _run_aco_vec(prizes, distance_matrix, max_len, current_eta)
    
    best_eta = current_eta.copy()
    best_score = base_score
    
    # Local Search / Coordinate Ascent
    # Step size for multiplicative updates
    step_up = 1.2
    step_down = 0.8
    
    # We can't iterate all 2500 entries multiple times if it's too slow.
    # Let's estimate cost of one ACO run.
    # If n=50, ants=20, iters=30.
    # 30 * 20 * (50^2 ops approx) = 30 * 20 * 2500 = 1.5M ops.
    # Python loop overhead for ants might be high.
    # Let's check time of one run.
    
    t0 = time.time()
    _run_aco_vec(prizes, distance_matrix, max_len, current_eta)
    t1 = time.time()
    time_per_eval = t1 - t0
    
    # Max evaluations allowed
    max_evals = int((deadline - time.time()) / time_per_eval)
    max_evals = max(10, max_evals) # At least 10 evals
    
    # We will perform random coordinate updates
    # Number of random coordinates to try per "epoch"
    # If max_evals is 1000, and we have 2500 coords, we can do 0.4 passes.
    # Let's just do random sampling of coordinates.
    
    # Use a fixed seed for the search to make it deterministic?
    # Or just use the current random state?
    # Since we restore state at the end, it doesn't matter for the ACO.
    # But for reproducibility of the heuristic finding, it might.
    # Let's just use np.random.
    
    for _ in range(max_evals):
        if time.time() > deadline:
            break
            
        # Pick random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        
        # Skip if i==j or prize[j] is 0 (low utility)
        if i == j or prizes[j] < 0.01:
            continue
            
        # Try up
        trial_eta = current_eta.copy()
        trial_eta[i, j] *= step_up
        trial_eta = np.maximum(trial_eta, 1e-9)
        
        score = _run_aco_vec(prizes, distance_matrix, max_len, trial_eta)
        
        if score > best_score:
            best_score = score
            best_eta = trial_eta
            current_eta = trial_eta
        else:
            # Try down
            trial_eta = current_eta.copy()
            trial_eta[i, j] *= step_down
            trial_eta = np.maximum(trial_eta, 1e-9)
            
            score = _run_aco_vec(prizes, distance_matrix, max_len, trial_eta)
            
            if score > best_score:
                best_score = score
                best_eta = trial_eta
                current_eta = trial_eta

    # Final validation
    best_eta = np.maximum(best_eta, 1e-9)
    
    # Restore random state
    np.random.set_state(rng_state)
    
    return best_eta
