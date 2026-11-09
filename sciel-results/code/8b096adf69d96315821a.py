import time
import sys

_LCM = 232792560

def _verify_recip_dissociated(subset):
    if not subset:
        return True
    k = len(subset)
    values = [_LCM // i for i in subset]
    sums = {0}
    for v in values:
        new_sums = {s + v for s in sums}
        if sums & new_sums:
            return False
        sums |= new_sums
    return True

def _can_add(sums, v):
    # Check if adding v causes collision
    # Collision if (s + v) in sums for any s in sums
    # Equivalent to: any(s + v in sums for s in sums)
    # Or: (sums + v) intersects sums
    # Fast check:
    # We can just iterate
    for s in sums:
        if s + v in sums:
            return False
    return True

def _backtrack_search(K, start_idx, current_set, current_sums, deadline, order):
    """
    Recursively try to add elements from 'order' starting at 'start_idx'
    to reach size K.
    """
    if len(current_set) == K:
        return current_set[:]
    
    if time.time() > deadline:
        return None
        
    # Pruning: not enough elements left
    if len(current_set) + (len(order) - start_idx) < K:
        return None

    for i in range(start_idx, len(order)):
        idx = order[i]
        v = _LCM // idx
        
        # Check if we can add idx
        if _can_add(current_sums, v):
            # Add it
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + v)
            
            # Recurse
            result = _backtrack_search(K, i + 1, current_set + [idx], new_sums, deadline, order)
            if result:
                return result
            # If no result, try next (backtrack)
        # If cannot add, skip (try next)
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(1, 23))
    
    # 1. Greedy Baseline
    # Use the logic from the current program to get a quick size 16 set
    # We'll implement a fast greedy here to ensure we have a fallback.
    
    def greedy(order):
        current_set = []
        current_sums = {0}
        for idx in order:
            v = _LCM // idx
            if _can_add(current_sums, v):
                current_set.append(idx)
                for s in list(current_sums):
                    current_sums.add(s + v)
        return current_set

    # Try a few greedy orderings to get a good baseline
    best_greedy = []
    best_greedy_size = 0
    
    # Order 1: Ascending
    s1 = greedy(candidates)
    if len(s1) > best_greedy_size:
        best_greedy = s1
        best_greedy_size = len(s1)
        
    # Order 2: Descending
    s2 = greedy(sorted(candidates, reverse=True))
    if len(s2) > best_greedy_size:
        best_greedy = s2
        best_greedy_size = len(s2)

    # Order 3: Random (fixed seed)
    import random
    rng = random.Random(42)
    for _ in range(10):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        s3 = greedy(shuffled)
        if len(s3) > best_greedy_size:
            best_greedy = s3
            best_greedy_size = len(s3)

    # Current best is likely 16.
    # Now try to find size 17 using backtracking.
    
    target_size = best_greedy_size + 1
    if target_size > 22:
        target_size = 22
        
    found = None
    # Try to find a set of target_size
    # Use multiple orderings for backtracking as well
    # Ascending order is good for symmetry breaking (indices must increase)
    # But the 'order' list determines which elements we consider.
    # We must ensure we consider ALL candidates, but in a specific sequence.
    # The backtracking function assumes 'order' contains the indices to try.
    # We should pass the full list of candidates, sorted, to ensure we don't miss combos.
    # However, the 'start_idx' logic implies we are picking a subsequence of 'order'.
    # So 'order' should be the full candidate list.
    
    search_deadline = start_time + 40.0 # Give 40s to backtracking
    
    # Try ascending
    res = _backtrack_search(target_size, 0, [], {0}, search_deadline, candidates)
    if res:
        found = res
        
    if not found and time.time() < search_deadline:
        # Try descending
        res = _backtrack_search(target_size, 0, [], {0}, search_deadline, sorted(candidates, reverse=True))
        if res:
            found = res

    if found:
        return found
    else:
        return best_greedy
