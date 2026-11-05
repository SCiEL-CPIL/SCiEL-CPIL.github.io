"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_fill_by_density(remaining_capacity, weights, values):
    """
    Fill knapsack greedily by value density.
    Returns the total value obtained.
    """
    if len(weights) == 0 or remaining_capacity <= 0:
        return 0.0
    
    # Compute densities
    densities = values / weights
    # Sort by density descending
    order = np.argsort(-densities)
    
    total_value = 0.0
    cap = remaining_capacity
    
    for idx in order:
        w = weights[idx]
        v = values[idx]
        if w <= cap:
            cap -= w
            total_value += v
        # else: item too heavy, skip (0/1 knapsack, can't take fraction)
    
    return total_value


def _greedy_fill_by_value(remaining_capacity, weights, values):
    """
    Fill knapsack greedily by value descending.
    Returns the total value obtained.
    """
    if len(weights) == 0 or remaining_capacity <= 0:
        return 0.0
    
    order = np.argsort(-values)
    
    total_value = 0.0
    cap = remaining_capacity
    
    for idx in order:
        w = weights[idx]
        v = values[idx]
        if w <= cap:
            cap -= w
            total_value += v
    
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.
    
    Strategy:
      1. Filter to feasible items (weight <= remaining_capacity).
      2. If none feasible, return -1.
      3. Use a 1-step look-ahead: for each feasible candidate, simulate picking
         it, then greedily filling the rest by density. Pick the candidate that
         yields the highest total simulated value.
      4. Time guard: if the look-ahead is running too long, fall back to pure
         density or value-based greedy.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Time budget: use at most ~0.5 seconds per call to stay well within 120s
    # for ~100 items * 8 instances = 800 calls max, so ~0.15s per call is safe
    # But actually, the evaluator calls this iteratively, so total time across
    # all calls must be < 120s. Let's use a small per-call budget.
    call_start = time.time()
    time_budget = 0.1  # 100ms per call, gives us plenty of margin
    
    # Get feasible weights and values
    f_weights = remaining_weights[feasible_indices]
    f_values = remaining_values[feasible_indices]
    f_densities = f_values / f_weights
    
    # Strategy: 1-step look-ahead with greedy density fill
    # For each feasible candidate, simulate picking it, then greedily fill
    best_score = -1.0
    best_idx = -1
    
    # If too many items, limit look-ahead to top candidates by density
    max_candidates = min(len(feasible_indices), 30)
    
    # Sort feasible indices by density descending for quick access
    density_order = np.argsort(-f_densities)
    top_candidates = density_order[:max_candidates]
    
    for candidate_pos in top_candidates:
        candidate_orig_idx = feasible_indices[candidate_pos]
        candidate_weight = f_weights[candidate_pos]
        candidate_value = f_values[candidate_pos]
        
        # Simulate: pick this item, then greedily fill the rest by density
        new_capacity = remaining_capacity - candidate_weight
        
        if new_capacity < 0:
            continue
        
        # Build remaining items (excluding the candidate)
        # We need to exclude candidate_pos from f_weights/f_values
        if len(f_weights) == 1:
            sim_value = candidate_value
        else:
            # Create mask to exclude candidate
            mask = np.ones(len(f_weights), dtype=bool)
            mask[candidate_pos] = False
            rem_weights = f_weights[mask]
            rem_values = f_values[mask]
            
            # Greedy fill by density
            if len(rem_weights) == 0:
                sim_value = candidate_value
            else:
                rem_densities = rem_values / rem_weights
                rem_order = np.argsort(-rem_densities)
                
                sim_value = candidate_value
                cap = new_capacity
                for idx in rem_order:
                    w = rem_weights[idx]
                    v = rem_values[idx]
                    if w <= cap:
                        cap -= w
                        sim_value += v
        
        if sim_value > best_score:
            best_score = sim_value
            best_idx = int(candidate_orig_idx)
        
        # Time check
        if time.time() - call_start > time_budget:
            break
    
    if best_idx != -1:
        return best_idx
    
    # Fallback: pick highest density feasible item
    return int(feasible_indices[np.argmax(f_densities)])


# EVOLVE-BLOCK-END
