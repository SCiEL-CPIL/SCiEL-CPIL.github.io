"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _find_max_dissociated():
    """
    Find the largest subset of [1, 22] such that all reciprocal subset sums are distinct.
    This is equivalent to finding a subset of indices {i} such that the set of integers {L/i}
    has all distinct subset sums.
    """
    # Precompute integer values v_i = L // i
    vals = {i: _L // i for i in range(1, 23)}
    
    def backtrack(start_idx, target_k, current_sums, chosen_indices):
        """
        Recursively try to add more indices to chosen_indices.
        start_idx: The next index to consider (to avoid duplicates and maintain order).
        target_k: The desired size of the subset.
        current_sums: Set of all subset sums of the currently chosen values.
        chosen_indices: List of indices chosen so far.
        Returns: A list of indices if a valid subset of size target_k is found, else None.
        """
        if len(chosen_indices) == target_k:
            return list(chosen_indices)
        
        # Pruning: Not enough remaining indices to reach target_k
        remaining_count = 22 - start_idx + 1
        if remaining_count < target_k - len(chosen_indices):
            return None
            
        # Iterate through candidates from start_idx to 22
        # Heuristic: Try smaller indices (larger values) first? 
        # Larger values tend to spread out sums, reducing collisions.
        # Let's try ascending index i (1 to 22).
        for i in range(start_idx, 23):
            v = vals[i]
            
            # Check for collision:
            # A collision exists if there are two distinct subsets of the current set
            # that have the same sum after adding v.
            # This happens if any existing sum s satisfies s + v == s' where s' is also an existing sum.
            # Since v > 0, s + v > s. We just need to check if s + v is in current_sums.
            has_collision = False
            for s in current_sums:
                if s + v in current_sums:
                    has_collision = True
                    break
            
            if has_collision:
                continue
                
            # No collision, proceed
            # Create new set of sums
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + v)
                
            # Recurse
            result = backtrack(i + 1, target_k, new_sums, chosen_indices + [i])
            if result is not None:
                return result
                
        return None

    # Try to find the maximum cardinality
    for k in range(22, 0, -1):
        # Start with empty set, sums = {0}
        result = backtrack(1, k, {0}, [])
        if result is not None:
            return result
            
    return [1] # Fallback, though k=1 always works

def construct():
    """Return a nonempty subset of [1,22] with all distinct reciprocal subset sums, maximizing cardinality."""
    return _find_max_dissociated()
# EVOLVE-BLOCK-END
