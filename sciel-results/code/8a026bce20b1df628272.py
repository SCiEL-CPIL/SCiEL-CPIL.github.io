"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 115.0  # safety margin before 120s kill
_START_TIME = None


def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _solve_fractional_knapsack(weights, values, capacity):
    """
    Solve the fractional knapsack relaxation.
    Returns the optimal fractional solution value and the set of items
    that are fully included (fraction = 1) in the optimal solution.
    The item with fraction in (0,1) is the 'pivot' item.
    
    For the heuristic, we want to pick the item that would be the NEXT one
    to be added in the fractional solution given the current remaining state.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, np.zeros(n, dtype=bool), -1
    
    # Compute value density
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    
    # Handle zero-weight items (take them all, infinite density)
    zero_weight_mask = (weights <= 0)
    if np.any(zero_weight_mask):
        # Take all zero-weight items
        pass
    
    # Sort by density descending
    # Items with zero weight get infinite density, so they come first
    # We need to be careful: for zero weight, density is inf
    order = np.argsort(-densities, kind='stable')
    
    remaining_cap = capacity
    full_items = np.zeros(n, dtype=bool)
    pivot_item = -1
    total_value = 0.0
    
    for idx in order:
        w = weights[idx]
        v = values[idx]
        
        if w <= 0:
            # Zero weight item, take it fully
            full_items[idx] = True
            total_value += v
            continue
        
        if remaining_cap <= 1e-12:
            break
        
        if w <= remaining_cap + 1e-12:
            # Take fully
            full_items[idx] = True
            total_value += v
            remaining_cap -= w
        else:
            # Take fractionally
            frac = remaining_cap / w
            total_value += v * frac
            pivot_item = idx
            remaining_cap = 0.0
            break
    
    return total_value, full_items, pivot_item


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic: Use the fractional knapsack relaxation as a surrogate.
    
    The idea: In the optimal fractional solution for the current remaining
    subproblem, items are taken in order of value/weight. The first item
    that is not yet in the solution (i.e., the one with the highest density
    that fits, or the pivot) is a good candidate for the next 0/1 choice.
    
    However, since we're building a 0/1 solution greedily, the best next
    item to add is the one with the highest value density among those that
    fit, BUT we should also consider that taking a high-density item that
    is small might leave room for more items.
    
    Actually, the fractional knapsack optimal solution tells us the ORDER
    in which items should be considered. The first item in that order that
    fits is the natural next choice. This is essentially the greedy density
    heuristic, but let's think if there's a refinement.
    
    Refinement: Instead of just picking the highest density item that fits,
    we can look at the fractional solution and pick the item that, when
    taken fully, results in the best continuation. But that's lookahead.
    
    For now, let's implement a slightly smarter version:
    1. Compute densities
    2. Among items that fit, pick the one with highest density
    3. But if there's a tie or close densities, prefer the one with higher
       absolute value (to pack more value per item)
    
    Actually, the key insight from the directive is to use the fractional
    knapsack optimum as a surrogate. The item that is the "next" item in
    the fractional solution (i.e., the highest density item that hasn't been
    taken yet and fits) is the natural choice.
    
    Let's also add a deadline guard.
    """
    elapsed = _get_elapsed()
    if elapsed > _DEADLINE_SECONDS:
        return -1
    
    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_values = np.asarray(remaining_values, dtype=np.float64)
    remaining_capacity = float(remaining_capacity)
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find items that fit
    fit_mask = remaining_weights <= remaining_capacity + 1e-12
    if not np.any(fit_mask):
        return -1
    
    # Among fitting items, compute densities
    fit_indices = np.where(fit_mask)[0]
    fit_weights = remaining_weights[fit_indices]
    fit_values = remaining_values[fit_indices]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        fit_densities = fit_values / fit_weights
    
    # Handle zero-weight items: they have infinite density
    zero_weight_in_fit = (fit_weights <= 0)
    if np.any(zero_weight_in_fit):
        # Take a zero-weight item (highest value among them)
        zero_vals = fit_values[zero_weight_in_fit]
        best_zero_idx = np.argmax(zero_vals)
        return int(fit_indices[zero_weight_in_fit][best_zero_idx])
    
    # Among positive-weight fitting items, pick highest density
    # Break ties by higher value
    max_density = np.max(fit_densities)
    # Find all items with density close to max
    close_to_max = np.isclose(fit_densities, max_density, rtol=1e-9, atol=1e-12)
    
    if np.any(close_to_max):
        candidates = fit_indices[close_to_max]
        candidate_values = fit_values[close_to_max]
        # Pick the one with highest value among those with max density
        best_local = np.argmax(candidate_values)
        return int(candidates[best_local])
    else:
        # Just pick the one with max density
        best_local = np.argmax(fit_densities)
        return int(fit_indices[best_local])
# EVOLVE-BLOCK-END
