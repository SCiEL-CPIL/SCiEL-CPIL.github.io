"""Program using a model-and-optimize approach for large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _fast_check_add(selected, x, sums_present):
    """
    Check if adding x to selected keeps all triple sums unique.
    Uses a bytearray for O(1) sum existence check.
    Returns (True, updated_sums) if valid, (False, None) otherwise.
    Assumes sums_present is a bytearray of size MAX_SUM + 1.
    """
    n = len(selected)
    # We need to check all triples (a, b, x) where a, b are from selected (with replacement)
    # and (x, x, x)
    
    # Optimization: 
    # 1. Check (x, x, x)
    # 2. Check (a, x, x) for a in selected
    # 3. Check (a, b, x) for a <= b in selected
    
    # We will collect new sums in a list first to avoid modifying sums_present prematurely
    # if a later sum collides.
    new_sums = []
    
    # (x, x, x)
    s = 3 * x
    if sums_present[s]:
        return (False, None)
    new_sums.append(s)
    
    # (a, x, x)
    for a in selected:
        s = a + 2 * x
        if sums_present[s]:
            return (False, None)
        # Check against other new_sums? 
        # Since a < x (if we maintain sorted order and x > max), then a+2x < x+2x=3x? No.
        # a+2x vs b+2x: distinct if a!=b.
        # a+2x vs 3x: a+2x=3x => a=x, impossible since distinct.
        # So (a,x,x) sums are distinct from each other and from (x,x,x).
        new_sums.append(s)
        
    # (a, b, x)
    for i in range(n):
        sa = selected[i]
        for j in range(i, n):
            sb = selected[j]
            s = sa + sb + x
            if sums_present[s]:
                return (False, None)
            # Check against new_sums?
            # sa+sb+x could collide with a'+2x?
            # sa+sb+x = a' + 2x => sa+sb - a' = x.
            # It's possible. So we must check against new_sums too.
            # However, checking against a growing list is O(K^2).
            # Given K is small (n^2/2 ~ 3000), and we do this for ~3000 sums, it's 9M ops.
            # Might be slow in Python.
            
            # Alternative: Just add to a temporary set?
            # Or, since we are checking against `sums_present` (old sums) first,
            # we only need to check for collisions among the NEW sums.
            # Let's use a local set for new sums to check mutual collisions.
            # But creating a set for every candidate is expensive.
            
            # Let's rely on the fact that if we pick x > max(selected), 
            # the sums a+b+x are generally larger than a+2x?
            # max(a+b+x) = 2*max(selected) + x.
            # max(a+2x) = max(selected) + 2x.
            # If x > max(selected), then 2x > max(selected).
            # So a+2x < max(selected) + 2x < 2x + 2x = 4x?
            # This doesn't guarantee disjoint ranges.
            
            # Let's just use a local set for new_sums to be safe and simple.
            # The overhead of creating a set of ~3000 ints is acceptable for ~1000 candidates.
            pass 
            
    # Re-implement with local set for new sums to ensure correctness
    # This is safer.
    
    return (False, None) # Placeholder, will rewrite

def _construct_randomized_greedy(limit, deadline, max_size=80):
    """
    Randomized greedy construction using a bytearray for sum tracking.
    """
    max_sum = 3 * limit
    sums_present = bytearray(max_sum + 1)
    selected = []
    
    # Candidate pool
    candidates = list(range(1, limit + 1))
    random.shuffle(candidates)
    
    # We want to add elements. To maximize cardinality, we should prefer smaller numbers?
    # Or random? Random is better to avoid bias.
    # But we must ensure we can fit 80.
    
    # Strategy: 
    # 1. Try to add a random candidate.
    # 2. If successful, remove it from pool.
    # 3. Repeat until pool empty or max_size reached.
    # 4. If we get stuck (no candidates work), try to remove a random element from selected 
    #    and retry with the remaining pool + removed element.
    
    attempts = 0
    max_attempts = 100000 # Safety
    
    while len(selected) < max_size and candidates and time.time() < deadline:
        added = False
        # Try a batch of random candidates
        for _ in range(min(100, len(candidates))):
            if not candidates:
                break
            idx = random.randint(0, len(candidates) - 1)
            x = candidates[idx]
            
            # Check if x can be added
            n = len(selected)
            new_sums_set = set()
            valid = True
            
            # (x, x, x)
            s = 3 * x
            if sums_present[s] or s in new_sums_set:
                valid = False
            else:
                new_sums_set.add(s)
            
            if valid:
                # (a, x, x)
                for a in selected:
                    s = a + 2 * x
                    if sums_present[s] or s in new_sums_set:
                        valid = False
                        break
                    new_sums_set.add(s)
            
            if valid:
                # (a, b, x)
                for i in range(n):
                    sa = selected[i]
                    for j in range(i, n):
                        sb = selected[j]
                        s = sa + sb + x
                        if sums_present[s] or s in new_sums_set:
                            valid = False
                            break
                        new_sums_set.add(s)
                    if not valid:
                        break
            
            if valid:
                # Accept x
                for s in new_sums_set:
                    sums_present[s] = 1
                selected.append(x)
                candidates.pop(idx)
                added = True
                break
        
        if not added:
            # Stuck. Try to remove a random element from selected to free up space.
            if selected:
                rem_idx = random.randint(0, len(selected) - 1)
                rem_val = selected[rem_idx]
                selected.pop(rem_idx)
                candidates.append(rem_val)
                
                # We need to rebuild sums_present? 
                # Removing an element is hard with a bytearray (need to know which sums to clear).
                # So, if we remove, we must rebuild sums_present from scratch.
                # This is expensive O(N^3).
                # Better strategy: Only do this if we have plenty of time.
                
                if time.time() < deadline - 10:
                    # Rebuild
                    sums_present = bytearray(max_sum + 1)
                    for x in selected:
                        # Add x to empty set? No, we removed one from a valid set.
                        # We need to re-add all selected elements.
                        # This is effectively restarting the greedy for the current set.
                        # Let's just restart the whole greedy with a different seed if we get stuck.
                        pass
            
            # Simpler: If stuck, break and restart with a new random seed?
            # Or try swapping: remove one, try to add two.
            # Given time constraints, let's just break and try a new randomization.
            break
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    limit = 4096
    max_size = 80
    
    best_result = []
    
    # Run multiple random greedy attempts
    for seed in range(100):
        random.seed(seed)
        result = _construct_randomized_greedy(limit, deadline, max_size)
        if len(result) > len(best_result):
            best_result = result
            if len(best_result) >= max_size:
                break
        
        if time.time() > deadline - 5:
            break
            
    return best_result

# EVOLVE-BLOCK-END
