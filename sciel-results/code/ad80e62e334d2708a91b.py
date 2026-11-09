import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_all(nums):
    if not nums:
        return 1
    return reduce(_lcm, nums)

def _is_valid(S):
    """Check if S is reciprocal-subsum-free."""
    if not S:
        return True
    
    S_list = sorted(S)
    L = _lcm_all(S_list)
    
    for x in S_list:
        target = L // x
        others = [y for y in S_list if y != x]
        if not others:
            continue
            
        vals = [L // y for y in others]
        total_sum = sum(vals)
        if total_sum < target:
            continue
            
        # DP for subset sum
        achievable = {0}
        for v in vals:
            if v > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return False
                
    return True

def _precompute_conflict_masks():
    """
    For each element x in [1,24], determine which subsets of OTHER elements
    can sum to 1/x. Return a list where mask[x] is a bitmask of elements
    that are "conflict partners" with x (i.e., could participate in forming 1/x).
    
    Actually, we need a more nuanced approach: two elements x and y are in conflict
    if y is part of SOME subset of other selected elements that sums to 1/x.
    But this depends on which other elements are selected.
    
    For branch-and-bound, we need: given a partial set P, can we add x?
    x is addable iff no subset of P sums to 1/x.
    
    We'll precompute for each x, the "required" elements: those y where 1/y <= 1/x
    (i.e., y >= x), since smaller reciprocals can't help sum to 1/x.
    Actually 1/y > 1/x when y < x, so y < x could contribute.
    
    Let's think differently: For each x, precompute all subsets T of [1,24]\{x}
    such that sum_{y in T} 1/y = 1/x. Then x conflicts with any element in T.
    But T must be a subset of the CURRENT selected set minus x.
    
    For efficiency, we'll store for each x, the set of "minimal" conflict sets.
    """
    N = 24
    # conflict_sets[x] = list of frozensets T where sum(1/y for y in T) = 1/x
    # and T does not contain x
    # But this can be exponential. Instead, we'll use a different approach.
    
    # For branch-and-bound, we need a fast check: given partial set P, is x addable?
    # This requires checking if any subset of P sums to 1/x.
    # We'll do this on-the-fly with memoization.
    
    pass

def _can_add(x, P):
    """Check if x can be added to set P (i.e., no subset of P sums to 1/x)."""
    if not P:
        return True
    
    P_list = [y for y in P if y != x]
    if not P_list:
        return True
    
    # Use LCM to convert to integer subset sum
    L = _lcm_all(P_list)
    target = L // x
    
    # Quick check: sum of all reciprocals in P_list
    total = sum(L // y for y in P_list)
    if total < target:
        return True  # Can't reach target
    
    # DP subset sum
    achievable = {0}
    for y in P_list:
        v = L // y
        if v > target:
            continue
        new_sums = set()
        for s in achievable:
            ns = s + v
            if ns <= target:
                new_sums.add(ns)
        achievable |= new_sums
        if target in achievable:
            return False
    
    return True

def _branch_and_bound(deadline):
    """
    Branch-and-bound to find maximum independent set.
    Uses bitset representation for efficiency.
    """
    N = 24
    # Represent sets as bitmasks (bit i set means element i+1 is in set)
    
    # Precompute: for each element x, which elements could be in a subset summing to 1/x?
    # An element y can contribute to 1/x only if 1/y <= 1/x, i.e., y >= x.
    # Actually no: 1/y > 1/x when y < x, so y < x CAN contribute (and more).
    # But if y > x, then 1/y < 1/x, so y could contribute if combined with others.
    
    # For each x, compute the "candidate pool": elements y != x that could be part of
    # a subset summing to 1/x. This is all y where 1/y <= 1/x (i.e., y >= x) OR
    # y < x (since 1/y > 1/x, y alone exceeds target, but could be part of a sum
    # with negative... no, all positive. So if 1/y > 1/x, then y ALONE exceeds target,
    # so y cannot be in any subset summing to 1/x.
    # Therefore: y can be in a subset summing to 1/x iff 1/y <= 1/x iff y >= x.
    
    # Wait, that's not right either. If y < x, then 1/y > 1/x. A subset containing y
    # has sum >= 1/y > 1/x, so it can't equal 1/x. Thus y < x CANNOT be in any subset
    # summing to 1/x.
    # So for target 1/x, only elements y >= x (and y != x) can participate.
    
    # This is a huge simplification!
    
    # For each x, candidate_elements[x] = {y : y > x, y in [1,24]}
    # (y = x is excluded since we consider subsets of OTHER elements)
    
    # Now, x is addable to P iff no subset of P ∩ {y > x} sums to 1/x.
    
    # Branch-and-bound state: (current_set, available_elements)
    # available_elements: bitmask of elements that can still be added
    
    best_size = 0
    best_set_mask = 0
    best_set_list = []
    
    # Order elements by some heuristic: process larger elements first?
    # Larger elements have fewer candidates (fewer y > x), so they're "easier" to add.
    # Smaller elements have more candidates, so they're "harder".
    # Process in order: 24, 23, 22, ..., 1
    
    # For each x, precompute the candidate mask: elements y > x
    candidate_mask = [0] * 25
    for x in range(1, 25):
        mask = 0
        for y in range(x + 1, 25):
            mask |= (1 << y)
        candidate_mask[x] = mask
    
    # Memoization: (x, P_mask) -> can_add(x, P)
    # But P_mask can be large. Let's limit memoization.
    memo = {}
    
    def _can_add_memo(x, P_mask):
        """Check if x can be added to set represented by P_mask."""
        key = (x, P_mask)
        if key in memo:
            return memo[key]
        
        # Extract elements from P_mask that are > x (only these can contribute)
        relevant_mask = P_mask & candidate_mask[x]
        if relevant_mask == 0:
            memo[key] = True
            return True
        
        # Get list of relevant elements
        P_list = []
        m = relevant_mask
        while m:
            lsb = m & (-m)
            y = lsb.bit_length() - 1
            P_list.append(y)
            m ^= lsb
        
        if not P_list:
            memo[key] = True
            return True
        
        L = _lcm_all(P_list)
        target = L // x
        
        total = sum(L // y for y in P_list)
        if total < target:
            memo[key] = True
            return True
        
        # DP subset sum
        achievable = {0}
        for y in P_list:
            v = L // y
            if v > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                memo[key] = False
                return False
        
        memo[key] = True
        return True
    
    def _upper_bound(available_mask):
        """Upper bound on number of additional elements we can add."""
        # Simple bound: count bits
        return available_mask.bit_count()
    
    def _dfs(current_mask, available_mask, depth):
        nonlocal best_size, best_set_mask, best_set_list
        
        if time.time() > deadline:
            return
        
        # Update best
        cur_size = current_mask.bit_count()
        if cur_size > best_size:
            best_size = cur_size
            best_set_mask = current_mask
            best_set_list = []
            m = current_mask
            while m:
                lsb = m & (-m)
                x = lsb.bit_length() - 1
                best_set_list.append(x)
                m ^= lsb
        
        # Pruning: if we can't beat best, return
        if cur_size + _upper_bound(available_mask) <= best_size:
            return
        
        # Choose next element to try
        # Heuristic: try elements in descending order (larger first, easier to add)
        # Or use a more sophisticated heuristic.
        
        m = available_mask
        while m:
            if time.time() > deadline:
                return
            
            lsb = m & (-m)
            x = lsb.bit_length() - 1
            m ^= lsb
            
            # Try adding x
            if _can_add_memo(x, current_mask):
                new_current = current_mask | (1 << x)
                # Remove x and all elements that conflict with x from available
                # An element y conflicts with x if y is in a subset of current_mask ∪ {x}
                # summing to 1/y. But this is complex.
                # Simpler: just remove x from available, and let the recursion handle conflicts.
                new_available = available_mask & ~(1 << x)
                _dfs(new_current, new_available, depth + 1)
            
            # Try not adding x (implicitly handled by continuing the loop)
        
    # Start DFS
    _dfs(0, (1 << 25) - 2, 0)  # Available: elements 1-24 (bits 1-24)
    
    if best_set_list:
        return sorted(best_set_list)
    return [24]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Try branch-and-bound first
    result = _branch_and_bound(deadline)
    
    # Verify result
    if result and _is_valid(result):
        return result
    
    # Fallback: greedy approach
    best_set = set()
    best_size = 0
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        current = set()
        candidates = list(range(1, 25))
        import random
        random.shuffle(candidates)
        
        for elem in candidates:
            if time.time() > deadline:
                break
            candidate = current | {elem}
            if _is_valid(candidate):
                current = candidate
        
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
    
    if best_set and _is_valid(best_set):
        return sorted(best_set)
    
    return [24]
