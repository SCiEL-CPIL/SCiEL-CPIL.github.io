import time
import math
import random

def _is_dissociated(S):
    """Check if set S has distinct reciprocal subset sums."""
    if not S:
        return True
    # Compute LCM
    L = 1
    for x in S:
        L = (L * x) // math.gcd(L, x)
    
    # Compute integer weights
    weights = [L // x for x in S]
    
    # Generate all subset sums
    # Using iterative doubling
    sums = [0]
    for w in weights:
        # Check for potential overflow? L for [1..22] is huge, but 
        # weights are L/x. 
        # LCM(1..22) is approx 232792560. Fits in int64? 
        # 2^31 is 2.1e9. 232M fits.
        # Sum of 12 such weights < 12 * 232M = 2.7e9. Might overflow int32, but Python int is arbitrary precision.
        # However, using Python lists of ints is slow.
        # We can use a set to track seen sums.
        # If we find a duplicate, return False immediately.
        pass
        
    # More efficient check:
    seen = {0}
    # This is still O(2^k) but allows early exit.
    # Actually, for small k, just generating the list and using set() is often faster in CPython 
    # due to C-level optimization of set construction, unless early exit is frequent.
    # Let's stick to the list generation + set comparison for simplicity and speed on valid sets.
    # But for invalid sets (collisions), early exit is better.
    
    # Hybrid:
    current_sums = [0]
    for w in weights:
        new_sums = [s + w for s in current_sums]
        # Check for collisions between current_sums and new_sums?
        # Actually, just check if len(current_sums + new_sums) == len(set(current_sums + new_sums))
        # This doubles the work each step.
        
        # Better:
        # If we maintain a set of seen sums:
        # seen = {0}
        # for w in weights:
        #   new_seen = set()
        #   for s in seen:
        #     ns = s + w
        #     if ns in seen or ns in new_seen: return False
        #     new_seen.add(ns)
        #   seen.update(new_seen)
        #   # Also need to check if any new_seen collides with previous seen?
        #   # The condition "all subset sums distinct" means the set of all sums has size 2^k.
        #   # This is equivalent to: for each step, the new sums (s+w) do not collide with existing sums.
        #   # Wait, if s1+w == s2 where s1, s2 in seen, that's a collision.
        #   # So:
        #   for s in seen:
        #       ns = s + w
        #       if ns in seen: return False
        #       # Also need to ensure ns doesn't collide with other new sums?
        #       # If s1 != s2, can s1+w == s2+w? No, w is constant.
        #       # So we just need to check if any s+w is in the *previous* seen set.
        #       # And add all new sums to seen.
        #   seen.update(s + w for s in seen) # Careful: iterating while updating? No, use a list.
        
    # Let's implement the efficient early-exit version.
    seen = {0}
    for w in weights:
        new_sums = []
        for s in seen:
            ns = s + w
            if ns in seen:
                return False
            new_sums.append(ns)
        # It is possible that two different s in seen produce the same ns?
        # No, s1 + w == s2 + w implies s1 == s2. Since s are unique in seen, ns are unique among themselves.
        # So we just need to check collision with existing 'seen'.
        seen.update(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    # Initial good set (size 9)
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    current_set = best_set[:]
    
    all_elements = list(range(1, 23))
    random.seed(42) # Reproducibility for testing, but maybe randomize? 
    # Better to use random without fixed seed for exploration, or multiple seeds.
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            no_improve_count = 0
            
        # 1. Try to add an element
        added = False
        remaining = [x for x in all_elements if x not in current_set]
        if remaining:
            # Shuffle to avoid bias
            random.shuffle(remaining)
            for cand in remaining:
                test_set = current_set + [cand]
                if _is_dissociated(test_set):
                    current_set = test_set
                    added = True
                    no_improve_count = 0
                    break # Restart search with new set
        
        if not added:
            # 2. Try swaps
            swapped = False
            in_set = set(current_set)
            remaining = [x for x in all_elements if x not in in_set]
            if remaining:
                random.shuffle(remaining)
                # Try swapping a random element in set with a random element outside
                # Limit attempts to avoid wasting time
                attempts = 0
                max_attempts = 50
                while attempts < max_attempts and time.time() < deadline:
                    attempts += 1
                    if not remaining or not current_set: break
                    out_cand = random.choice(remaining)
                    in_cand = random.choice(current_set)
                    
                    if out_cand == in_cand: continue
                    
                    # Create new set
                    new_set = [x for x in current_set if x != in_cand] + [out_cand]
                    
                    if _is_dissociated(new_set):
                        # Accept swap if it keeps size same (to explore) 
                        # We don't update best_set here as size didn't increase
                        current_set = new_set
                        # Update remaining list
                        in_set = set(current_set)
                        remaining = [x for x in all_elements if x not in in_set]
                        swapped = True
                        break
            
            if not swapped:
                no_improve_count += 1
                # If stuck for a while, reset to best or random restart
                if no_improve_count > 200:
                    # Reset to best to try different additions
                    current_set = best_set[:]
                    no_improve_count = 0
                    # Or try a random restart of size len(best_set)
                    if random.random() < 0.1:
                        # Random restart
                        k = len(best_set)
                        current_set = random.sample(all_elements, k)
                        if _is_dissociated(current_set):
                             pass # Valid start
                        else:
                             current_set = best_set[:] # Fallback

    return sorted(best_set)
