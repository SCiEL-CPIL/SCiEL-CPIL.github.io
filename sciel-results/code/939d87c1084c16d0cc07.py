"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    all_elements = list(range(1, 25))
    N = 24  # elements 1..24
    
    # Precompute LCM of all elements
    global_lcm = _lcm_list(all_elements)
    
    # Convert each element x to weight = global_lcm // x
    # So 1/x = weight / global_lcm
    weights = [global_lcm // x for x in all_elements]
    
    # For each element i (0-indexed), precompute the set of all subset sums
    # of weights of OTHER elements, using meet-in-the-middle
    # But that's expensive for all 24 elements. Instead, use a faster approach:
    # For each target i, we need to check if weights[i] can be formed as sum of
    # a subset of {weights[j] : j != i}
    
    # Precompute all subset sums using bitmasks for the full set, then for each
    # target, check if the target weight is in the subset sums of others.
    
    # More efficient: for each i, compute all subset sums of weights excluding i
    # using meet-in-the-middle, and check if weights[i] is achievable.
    
    # Precompute conflict structure:
    # conflict[i] = set of subsets S of [0..23]\{i} such that sum_{j in S} weights[j] == weights[i]
    # We only need to know if such a subset exists (for validity) and which elements are involved.
    
    # For each i, determine if there's a violation and which elements are in the violating subset
    # This is the key precomputation step.
    
    def _find_violating_subset(i, active_mask):
        """
        For target element i, check if weights[i] can be formed as sum of a subset
        of weights of elements in active_mask (excluding i).
        Returns (bool, bitmask_of_one_violating_subset) or (False, 0).
        """
        target = weights[i]
        # Get active elements excluding i
        others_mask = active_mask & ~(1 << i)
        if others_mask == 0:
            return (False, 0)
        
        # Extract list of weights for active others
        others_weights = []
        others_indices = []
        for j in range(N):
            if (others_mask >> j) & 1:
                others_weights.append(weights[j])
                others_indices.append(j)
        
        m = len(others_weights)
        if m == 0:
            return (False, 0)
        
        # Meet in the middle
        half = m // 2
        left_w = others_weights[:half]
        right_w = others_weights[half:]
        left_idx = others_indices[:half]
        right_idx = others_indices[half:]
        
        # Generate all subset sums of left
        left_sums = {}  # sum -> bitmask of indices (within left)
        left_sums[0] = 0
        for mask in range(1, 1 << len(left_w)):
            s = 0
            idx_mask = 0
            for b in range(len(left_w)):
                if mask & (1 << b):
                    s += left_w[b]
                    idx_mask |= (1 << b)
            if s not in left_sums:
                left_sums[s] = idx_mask
        
        # Generate all subset sums of right
        right_sums = {}  # sum -> bitmask of indices (within right)
        right_sums[0] = 0
        for mask in range(1, 1 << len(right_w)):
            s = 0
            idx_mask = 0
            for b in range(len(right_w)):
                if mask & (1 << b):
                    s += right_w[b]
                    idx_mask |= (1 << b)
            if s not in right_sums:
                right_sums[s] = idx_mask
        
        # Check if target can be formed
        for rs, ridx in right_sums.items():
            if rs > target:
                continue
            ls_needed = target - rs
            if ls_needed in left_sums:
                lidx = left_sums[ls_needed]
                # Build the full bitmask
                full_mask = 0
                for b in range(len(left_idx)):
                    if lidx & (1 << b):
                        full_mask |= (1 << left_idx[b])
                for b in range(len(right_idx)):
                    if ridx & (1 << b):
                        full_mask |= (1 << right_idx[b])
                return (True, full_mask)
        
        return (False, 0)
    
    # Precompute for each element i, the set of "conflict partners"
    # An element j is a conflict partner of i if j appears in some violating subset for i
    # But more precisely, we need: for the current active set, does i have a violation?
    
    # Strategy: Start with all elements active. Iteratively remove elements that have violations.
    # When we remove an element, it may resolve violations for others.
    
    active_mask = (1 << N) - 1  # all 24 elements active
    
    # Iterative pruning
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            if not (active_mask >> i) & 1:
                continue
            has_violation, viol_mask = _find_violating_subset(i, active_mask)
            if has_violation:
                # Remove element i (the target)
                active_mask &= ~(1 << i)
                changed = True
    
    # Convert active_mask to list
    current = [i + 1 for i in range(N) if (active_mask >> i) & 1]
    
    # Try to add back removed elements
    removed = [x for x in all_elements if x not in current]
    for x in removed:
        if time.time() > deadline - 3:
            break
        idx = x - 1
        trial_mask = active_mask | (1 << idx)
        # Check if adding x creates any violation
        valid = True
        for j in range(N):
            if time.time() > deadline - 3:
                valid = False
                break
            if not (trial_mask >> j) & 1:
                continue
            has_violation, _ = _find_violating_subset(j, trial_mask)
            if has_violation:
                valid = False
                break
        if valid:
            active_mask = trial_mask
            current = [i + 1 for i in range(N) if (active_mask >> i) & 1]
    
    # Final greedy improvement: try swapping
    if time.time() < deadline - 3:
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 3:
                break
            removed = [x for x in all_elements if x not in current]
            for x in removed[:]:
                if time.time() > deadline - 3:
                    break
                idx = x - 1
                trial_mask = active_mask | (1 << idx)
                valid = True
                for j in range(N):
                    if time.time() > deadline - 3:
                        valid = False
                        break
                    if not (trial_mask >> j) & 1:
                        continue
                    has_violation, _ = _find_violating_subset(j, trial_mask)
                    if has_violation:
                        valid = False
                        break
                if valid:
                    active_mask = trial_mask
                    current = [i + 1 for i in range(N) if (active_mask >> i) & 1]
                    improved = True
    
    if not current:
        current = [24]
    
    return current
# EVOLVE-BLOCK-END
