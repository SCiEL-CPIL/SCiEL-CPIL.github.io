"""Capacitated facility location: vectorized heuristic + local search with time budget."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _true_obj(result, demands, capacities, open_cost, assign_cost):
    """Compute objective: open costs + assignment costs. Returns inf if infeasible."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        return float('inf')
    open_sum = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_sum += open_cost[f]
    assign_sum = float(np.sum(assign_cost[np.arange(len(result)), result]))
    return open_sum + assign_sum


def _vectorized_seed(demands, capacities, assign_cost):
    """Vectorized heuristic: assign all customers to their cheapest facility, then repair capacity violations."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Start with each customer assigned to their cheapest facility
    result = np.argmin(assign_cost, axis=1)
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]
    
    # Repair: move customers from overloaded facilities to others
    # Process facilities in order of most overloaded first
    for _ in range(n_fac * 3):  # Max iterations for repair
        # Find facilities with excess load
        excess_mask = load > capacities
        if not np.any(excess_mask):
            break
        
        # For each overloaded facility, move customers with highest demand to other facilities
        for f in np.where(excess_mask)[0]:
            while load[f] > capacities[f]:
                # Find customers at facility f
                customers_at_f = np.where(result == f)[0]
                if len(customers_at_f) == 0:
                    break
                
                # Pick the customer with highest demand (easiest to move)
                c = customers_at_f[np.argmax(demands[customers_at_f])]
                d = demands[c]
                
                # Find feasible alternative facilities sorted by cost
                # Exclude current facility
                alt_costs = assign_cost[c].copy()
                alt_costs[f] = np.inf
                alt_order = np.argsort(alt_costs)
                
                moved = False
                for af in alt_order:
                    if load[af] + d <= capacities[af]:
                        # Move customer
                        load[f] -= d
                        load[af] += d
                        result[c] = af
                        moved = True
                        break
                
                if not moved:
                    # Cannot fix this violation; break to avoid infinite loop
                    break
    
    # Final feasibility check - if still infeasible, use greedy as fallback
    if np.any(load > capacities):
        return _greedy_seed(demands, capacities, assign_cost)
    
    return result


def _greedy_seed(demands, capacities, assign_cost):
    """Largest-first greedy assignment (fallback)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        feasible = []
        for f in np.argsort(assign_cost[i]):
            if load[f] + d <= capacities[f]:
                feasible.append(f)
        if feasible:
            result[i] = feasible[0]
            load[result[i]] += d
        else:
            result[i] = 0
            load[0] += d
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Local search: vectorized single moves + pair swaps."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    
    best_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
    if best_obj == float('inf'):
        return result, best_obj
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # --- Single Moves (Vectorized Selection) ---
        # Calculate potential delta for all (c, f) pairs
        # delta[c, f] = assign_cost[c, f] - assign_cost[c, result[c]] + open_cost_delta
        # We need to be careful with open cost delta.
        
        # Current costs
        cur_costs = assign_cost[np.arange(n_cust), result]
        
        # Base assignment cost difference
        # diff[c, f] = assign_cost[c, f] - cur_costs[c]
        diff = assign_cost - cur_costs[:, np.newaxis]
        
        # Mask out current facility
        for c in range(n_cust):
            diff[c, result[c]] = np.inf
            
        # Mask out infeasible facilities (capacity)
        # feasible[c, f] = (load[f] + demands[c] <= capacities[f])
        # load is (F,), demands is (N,)
        # new_loads = load[np.newaxis, :] + demands[:, np.newaxis]
        new_loads = load[np.newaxis, :] + demands[:, np.newaxis]
        feasible = new_loads <= capacities[np.newaxis, :]
        
        # Set infeasible to inf
        diff[~feasible] = np.inf
        
        # Open cost adjustments
        # This is the tricky part to vectorize perfectly because it depends on 
        # whether the source becomes empty and target becomes non-empty.
        # 
        # If we move c from f_c to f:
        # Source f_c becomes empty if load[f_c] == demands[c].
        # Target f becomes non-empty if load[f] == 0.
        
        # Let's compute the open cost delta matrix
        # open_delta[c, f] = -open_cost[result[c]] if load[result[c]] == demands[c] else 0
        #                   + open_cost[f] if load[f] == 0 else 0
        
        # We can approximate this or calculate it.
        # Since n_fac is small (20), we can loop over facilities for open costs?
        # Or vectorize:
        
        # Condition: source becomes empty
        # src_empty = (load[result] == demands) # (N,)
        # If src_empty[c], we save open_cost[result[c]]
        
        # Condition: target becomes non-empty
        # tgt_empty = (load == 0) # (F,)
        # If tgt_empty[f], we pay open_cost[f]
        
        # open_delta[c, f] = (src_empty[c] * -open_cost[result[c]]) + (tgt_empty[f] * open_cost[f])
        
        src_empty = (load[result] == demands)
        tgt_empty = (load == 0)
        
        # Add open cost effects
        # Note: if source and target are same, we masked it out.
        # If source becomes empty, we subtract open_cost[result[c]] for all f?
        # Yes, if we move c out of result[c], and it was the only one, we save the cost.
        # This saving applies regardless of where we move c (as long as we move).
        
        open_cost_src = np.where(src_empty, -open_cost[result], 0.0) # (N,)
        open_cost_tgt = np.where(tgt_empty, open_cost, 0.0) # (F,)
        
        # delta_matrix = diff + open_cost_src[:, np.newaxis] + open_cost_tgt[np.newaxis, :]
        
        delta_matrix = diff + open_cost_src[:, np.newaxis] + open_cost_tgt[np.newaxis, :]
        
        # Find best move for each customer
        best_f = np.argmin(delta_matrix, axis=1)
        best_delta = np.min(delta_matrix, axis=1)
        
        # We want to apply moves with delta < 0.
        # But we must ensure capacity is still valid after applying.
        # Since we selected best_f based on current loads, applying one move changes loads.
        # We can sort customers by best_delta (most negative first) and apply if still feasible.
        
        candidates = np.where(best_delta < -1e-9)[0]
        # Sort by delta (ascending, so most negative first)
        candidates = candidates[np.argsort(best_delta[candidates])]
        
        for c in candidates:
            if time.time() >= deadline:
                break
            f_new = best_f[c]
            f_old = result[c]
            d = demands[c]
            
            # Re-check capacity (it might have changed if another move to f_new happened)
            if load[f_new] + d <= capacities[f_new]:
                # Apply move
                delta = best_delta[c]
                # Update loads
                load[f_old] -= d
                load[f_new] += d
                result[c] = f_new
                best_obj += delta
                improved = True
                no_improve = 0
                # We could break here to restart the vectorization, but let's try to apply multiple
                # However, the delta_matrix is now stale.
                # For simplicity and correctness, we should probably break after one move
                # or recompute. Recomputing is O(N*F) which is fast.
                # Let's break to recompute for next iteration.
                break
        
        if not improved:
            no_improve += 1
            
            # --- Pair Swaps ---
            # Only do swaps if no single move improved.
            # Or do swaps periodically.
            
            # We will try to find a beneficial swap.
            # Iterate c1, c2.
            # To speed up, we can limit to customers in different facilities.
            
            # We can also vectorize the swap check?
            # For each pair (c1, c2), f1=result[c1], f2=result[c2].
            # New load f1: load[f1] - d1 + d2
            # New load f2: load[f2] - d2 + d1
            # Check caps.
            # Delta = (A[c1,f2] + A[c2,f1] - A[c1,f1] - A[c2,f2]) + open_delta
            
            # Since N=100, N^2=10000. This is fast enough to loop in Python if we don't do it every single iteration.
            # Let's do it if no_improve > 0.
            
            swap_improved = False
            # Optimization: only consider pairs where facilities are different
            # We can just loop.
            for c1 in range(n_cust):
                if time.time() >= deadline:
                    break
                f1 = result[c1]
                d1 = demands[c1]
                for c2 in range(c1 + 1, n_cust):
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d2 = demands[c2]
                    
                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1
                    
                    if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                        continue
                    
                    # Calculate delta
                    delta_assign = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                                   (assign_cost[c1, f1] + assign_cost[c2, f2])
                    
                    # Open cost delta
                    delta_open = 0.0
                    # f1 changes from load[f1] to new_load_f1
                    # f2 changes from load[f2] to new_load_f2
                    
                    if load[f1] > 0 and new_load_f1 == 0:
                        delta_open -= open_cost[f1]
                    elif load[f1] == 0 and new_load_f1 > 0:
                        delta_open += open_cost[f1]
                        
                    if load[f2] > 0 and new_load_f2 == 0:
                        delta_open -= open_cost[f2]
                    elif load[f2] == 0 and new_load_f2 > 0:
                        delta_open += open_cost[f2]
                    
                    delta = delta_assign + delta_open
                    
                    if delta < -1e-9:
                        result[c1], result[c2] = f2, f1
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        best_obj += delta
                        improved = True
                        swap_improved = True
                        no_improve = 0
                        break
                if swap_improved:
                    break
            
            if not swap_improved and not improved:
                no_improve += 1
                # If we do nothing, we might loop forever. 
                # But no_improve counter handles termination.
                # To avoid infinite loop on stuck state, we can add a small random perturbation or just break.
                # Given the time budget, breaking is safer.
                if no_improve > 10:
                    break
        else:
            no_improve = 0
    
    return result, best_obj


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using vectorized seed + local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 145s max
    deadline = time.time() + 145.0
    
    # Vectorized seed (cheapest facility + capacity repair)
    result = _vectorized_seed(demands, capacities, assign_cost)
    
    # Local search with time budget
    result, best_obj = _local_search(result, demands, capacities, open_cost, assign_cost, deadline)
    
    # Verify feasibility
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        # Fallback: return greedy result
        return _greedy_seed(demands, capacities, assign_cost)
    
    return result.astype(int)
# EVOLVE-BLOCK-END
