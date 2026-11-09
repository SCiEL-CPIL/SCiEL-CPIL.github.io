import numpy as np
import time

# Adaptive parameters
# We maintain a small history of items to estimate typical size
_HISTORY_SIZE = 100
_item_history = []
_typical_item_size = 1.0  # Initial guess
_last_update_time = time.time()

_INFEASIBLE = -1e9

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin based on a 'waste-avoidance' lookahead heuristic.
    We penalize bins that leave a remaining capacity in a 'dead zone'
    where it is unlikely to fit future items, thereby reducing fragmentation.
    """
    global _item_history, _typical_item_size, _last_update_time
    
    # Update typical item size periodically to avoid overhead on every call
    # This is a heuristic to keep the function fast
    current_time = time.time()
    if current_time - _last_update_time > 0.01: # Update roughly every 10ms
        _item_history.append(item)
        if len(_item_history) > _HISTORY_SIZE:
            _item_history.pop(0)
        _typical_item_size = np.mean(_item_history) if _item_history else item
        _last_update_time = current_time
    else:
        # Still add to history but don't compute mean every time if too frequent?
        # For simplicity and safety, just add. The mean calculation is cheap for N=100.
        _item_history.append(item)
        if len(_item_history) > _HISTORY_SIZE:
            _item_history.pop(0)
        # We could update typical size here too, it's O(1) amortized if we use a running sum, 
        # but np.mean on 100 floats is negligible. Let's just do it every call for accuracy 
        # but keep the time check for potential future complex updates.
        _typical_item_size = np.mean(_item_history)

    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    # Base score: Prefer tighter fit (smaller remaining)
    # We use -remaining so that 0 remaining is high score.
    score = -remaining
    
    # Lookahead Waste Penalty:
    # If the remaining space is in a "dead zone", it's effectively wasted.
    # Dead zone definition: 
    #   Lower bound: item * 0.1 (too small to be useful for most items)
    #   Upper bound: item * 0.5 (likely too small for next item if items are similar)
    # We estimate the next item size to be close to the typical size.
    
    # Scale the dead zone based on the typical item size, not just the current one,
    # but the current item is a good proxy for the local distribution.
    # Let's use the current item size to define the zone, as it's the most immediate context.
    
    dead_lower = 0.1 * item
    dead_upper = 0.5 * item
    
    # Create a mask for bins that fall into the dead zone
    in_dead_zone = (remaining > dead_lower) & (remaining < dead_upper) & feasible
    
    # Apply a strong penalty to these bins
    # The penalty should be enough to make them less attractive than a slightly looser fit
    # that leaves either very small waste (< dead_lower) or very large waste (> dead_upper)
    # which might fit another item.
    penalty = 5.0 * item  # Strong penalty
    
    score = score - np.where(in_dead_zone, penalty, 0.0)
    
    # Consolidation bias: Still prefer bins that are already used to some extent
    # to reduce the number of active bins, but secondary to waste avoidance.
    # The waste avoidance already encourages filling bins to near capacity,
    # so this might be redundant or even counter-productive if it pushes into dead zones.
    # However, a small bias helps break ties in favor of "fuller" bins which are 
    # more likely to be closed soon.
    # Let's keep a very small consolidation bias.
    score = score + 0.05 * bins
    
    # Mask infeasible bins
    score = np.where(feasible, score, _INFEASIBLE)
    
    return score
