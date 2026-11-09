"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Option-value heuristic for ACO orienteering.
    
    For each transition i->j, the heuristic reflects:
      1. The immediate prize of node j
      2. The option value: expected prize from nodes still reachable
         after spending d(i,j) + d(j,0) of the budget
    
    This models the full trajectory value rather than just local distance cost.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    eps = 1e-9
    
    d_j0 = distance_matrix[:, 0]  # shape (n,)
    
    # Cost to go from i to j and back to depot: d(i,j) + d(j,0)
    # This is the minimum budget consumed by visiting j from i
    transition_cost = distance_matrix + d_j0[None, :]  # shape (n, n)
    
    # For each node j, compute the "option value": for a given remaining budget R,
    # what is the expected prize we can collect from other nodes?
    # After visiting j from i, remaining budget = max_len - transition_cost[i,j]
    # We can visit node k (k != j, k != 0) if d(j,k) + d(k,0) <= remaining_budget
    
    # Precompute: for each node j, and for a given budget R,
    # sum of prizes of nodes k such that d(j,k) + d(k,0) <= R
    # This is a 2D lookup: option_value[j, R]
    
    # We'll compute this discretely. max_len is small (3.0), distances are in [0, ~1.4]
    # Let's compute for each pair (j, k): cost_jk0 = d(j,k) + d(k,0)
    cost_jk0 = distance_matrix + d_j0[None, :]  # shape (n, n), cost_jk0[j,k] = d(j,k)+d(k,0)
    
    # For each node j, sort other nodes by their cost to visit from j
    # option_value[j, R] = sum of prizes[k] for k where cost_jk0[j,k] <= R
    
    # Since n=50, we can do this efficiently
    # For each j, create a sorted list of (cost, prize) for all k != j, k != 0
    # Then for any R, option_value is the cumulative sum up to R
    
    # Build option value function for each node j
    # option_vals[j, r] where r is a discretized budget level
    # Use fine discretization of [0, max_len]
    n_bins = 100
    budget_levels = np.linspace(0, max_len, n_bins)
    
    # option_val[j, b] = sum of prizes[k] for k != j, k != 0, with cost_jk0[j,k] <= budget_levels[b]
    option_val = np.zeros((n, n_bins))
    
    for j in range(n):
        if j == 0:
            continue
        costs = cost_jk0[j, :].copy()
        prizes_k = prizes.copy()
        # Exclude j itself and node 0
        costs[j] = np.inf
        costs[0] = np.inf
        
        # Sort by cost
        sorted_idx = np.argsort(costs)
        sorted_costs = costs[sorted_idx]
        sorted_prizes = prizes_k[sorted_idx]
        
        # Cumulative sum
        cum_prizes = np.cumsum(sorted_prizes)
        
        # For each budget level, find how many nodes fit
        for b in range(n_bins):
            R = budget_levels[b]
            count = np.searchsorted(sorted_costs, R, side='right')
            if count > 0:
                option_val[j, b] = cum_prizes[count - 1]
    
    # Now compute the heuristic
    # eta[i,j] = prize[j] + alpha * option_value(j, remaining_budget)
    # where remaining_budget = max_len - transition_cost[i,j]
    # But we need to handle infeasible cases
    
    eta = np.full((n, n), eps)
    
    alpha = 0.5  # weight for future option value
    
    for i in range(n):
        for j in range(n):
            if i == j or j == 0:
                eta[i, j] = eps
                continue
            
            tc = transition_cost[i, j]
            if tc > max_len:
                eta[i, j] = eps
                continue
            
            remaining = max_len - tc
            if remaining < 0:
                eta[i, j] = eps
                continue
            
            # Immediate prize
            immediate = prizes[j]
            
            # Option value: interpolate from budget_levels
            if remaining <= 0:
                ov = 0.0
            elif remaining >= max_len:
                ov = option_val[j, n_bins - 1]
            else:
                # Find the bin
                idx = np.searchsorted(budget_levels, remaining, side='right') - 1
                if idx < 0:
                    ov = 0.0
                elif idx >= n_bins - 1:
                    ov = option_val[j, n_bins - 1]
                else:
                    # Linear interpolation
                    frac = (remaining - budget_levels[idx]) / (budget_levels[idx + 1] - budget_levels[idx])
                    ov = option_val[j, idx] * (1 - frac) + option_val[j, idx + 1] * frac
            
            eta[i, j] = immediate + alpha * ov
    
    # Ensure positive and finite
    eta = np.maximum(eta, eps)
    eta = np.where(np.isfinite(eta), eta, eps)
    
    return eta
# EVOLVE-BLOCK-END
