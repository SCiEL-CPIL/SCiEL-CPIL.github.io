"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _search_max_cardinality():
    """Find the largest subset of [1,22] with all distinct reciprocal subset sums."""
    candidates = list(range(1, 23))
    # Sort by a ascending (1, 2, ..., 22) which corresponds to values L/a descending.
    # Larger values first are generally "safer" for distinct subset sums.
    candidates.sort()
    
    # Precompute integer values
    vals = {a: _L // a for a in candidates}
    
    # Try from largest cardinality down
    for k in range(22, 0, -1):
        # Backtracking search with incremental subset sum checking
        result = _backtrack(candidates, vals, k, 0, {0}, [])
        if result is not None:
            return result
    return [1]  # fallback

def _backtrack(candidates, vals, target_k, idx, current_sums, current_subset):
    """
    Backtrack to find a valid subset of size target_k from candidates starting at idx.
    current_sums: set of all subset sums of current_subset (scaled by LCM)
    """
    # Pruning: not enough candidates left to reach target_k
    if len(candidates) - idx < target_k - len(current_subset):
        return None
        
    # Base case: found target size
    if len(current_subset) == target_k:
        return current_subset
        
    # Option 1: Include candidates[idx]
    a = candidates[idx]
    v = vals[a]
    
    # Check if adding v causes collisions with existing sums
    # New sums are {s + v for s in current_sums}
    # Collision if (s + v) is in current_sums for any s in current_sums
    # Internal collisions in new sums are impossible since v > 0 and s are distinct.
    has_collision = False
    for s in current_sums:
        if (s + v) in current_sums:
            has_collision = True
            break
            
    if not has_collision:
        # Build new set of sums
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + v)
            
        result = _backtrack(candidates, vals, target_k, idx + 1, new_sums, current_subset + [a])
        if result is not None:
            return result

    # Option 2: Exclude candidates[idx]
    result = _backtrack(candidates, vals, target_k, idx + 1, current_sums, current_subset)
    if result is not None:
        return result

    return None

def construct():
    """Return a nonempty subset of [1,22] with all distinct reciprocal subset sums, maximizing cardinality."""
    result = _search_max_cardinality()
    return result
# EVOLVE-BLOCK-END
