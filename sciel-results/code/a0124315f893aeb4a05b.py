import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    start_time = time.time()
    deadline = start_time + 140.0  # 140s margin out of 150s total

    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    C = float(remaining_capacity)

    if C <= 0:
        return -1

    # Filter feasible items
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    n_f = len(w_f)

    if n_f == 0:
        return -1

    # Check for zero-weight items with positive value
    zero_w_mask = (w_f == 0) & (v_f > 0)
    if np.any(zero_w_mask):
        zero_indices = np.where(zero_w_mask)[0]
        return int(feasible_indices[zero_indices[0]])

    # Compute densities for initial candidate ranking
    d_f = np.zeros(n_f)
    non_zero_w = w_f > 0
    d_f[non_zero_w] = v_f[non_zero_w] / w_f[non_zero_w]
    d_f[~non_zero_w] = 0.0

    # Sort by density descending to get top candidates
    sort_idx = np.argsort(-d_f)
    
    # Determine how many candidates to evaluate with MCTS
    # Start with top 20 by density, can expand if time permits
    n_candidates = min(20, n_f)
    candidate_indices = sort_idx[:n_candidates]
    
    # Helper: simulate greedy completion after picking a specific item first
    # Returns total value of the complete solution
    def simulate_completion(pick_idx, capacity, weights, values, rng):
        """
        Pick item pick_idx first, then greedily fill remaining capacity
        using random perturbation of density order.
        """
        cap = capacity - weights[pick_idx]
        total_val = values[pick_idx]
        
        if cap <= 1e-9:
            return total_val
        
        # Create list of remaining items (excluding pick_idx)
        remaining = [i for i in range(len(weights)) if i != pick_idx and weights[i] <= cap]
        
        if not remaining:
            return total_val
        
        # For each simulation, randomly perturb the order
        # Use a mix of density-based and random selection
        sim_count = 50  # number of random simulations per candidate
        
        total_sim_value = 0.0
        
        for _ in range(sim_count):
            cur_cap = cap
            sim_val = total_val
            
            # Randomly shuffle remaining items with density bias
            # Higher density items have higher probability of being selected earlier
            probs = np.array([d_f[i] + 0.1 for i in remaining])
            probs = probs / probs.sum()
            
            # Sample without replacement
            perm = rng.choice(len(remaining), size=len(remaining), replace=False, p=probs)
            
            for pos in perm:
                idx = remaining[pos]
                if weights[idx] <= cur_cap:
                    sim_val += values[idx]
                    cur_cap -= weights[idx]
                    if cur_cap <= 1e-9:
                        break
            
            total_sim_value += sim_val
        
        return total_sim_value / sim_count
    
    # MCTS-style evaluation
    rng = np.random.default_rng(42)
    
    best_score = -1.0
    best_i_local = -1
    
    # Evaluate each candidate
    for cand_local in candidate_indices:
        if time.time() >= deadline:
            break
        
        # Simulate completion if we pick this item first
        score = simulate_completion(cand_local, C, w_f, v_f, rng)
        
        if score > best_score:
            best_score = score
            best_i_local = cand_local
    
    # If MCTS didn't find anything (shouldn't happen), fall back to density greedy
    if best_i_local == -1:
        best_i_local = candidate_indices[0]
    
    return int(feasible_indices[best_i_local])
