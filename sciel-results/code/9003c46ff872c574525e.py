import time
import math
import random

def _is_dissociated(S):
    """Check if set S has distinct reciprocal subset sums using integer arithmetic."""
    if not S:
        return True
    
    # Compute LCM of all elements in S
    L = 1
    for x in S:
        L = (L * x) // math.gcd(L, x)
    
    # Convert to integer weights
    weights = [L // x for x in S]
    
    n = len(weights)
    if n > 22:
        return False # Safety, shouldn't happen
    
    # Efficient check using set of seen sums
    # For n=16, 2^16 = 65536 sums. This is very fast in Python.
    seen = {0}
    for w in weights:
        new_sums = []
        # We iterate over the *current* state of 'seen'
        # To avoid "dictionary changed size during iteration", we iterate over a list copy
        # or we build new_sums first.
        # Since s1+w == s2+w implies s1==s2, new_sums are unique among themselves.
        # We only need to check collision with existing 'seen'.
        for s in seen:
            ns = s + w
            if ns in seen:
                return False
            new_sums.append(ns)
        seen.update(new_sums)
    
    return True

def _try_add(S, x):
    """Try to add x to S, return new set if dissociated, else None."""
    if x in S:
        return S
    new_S = S + [x]
    if _is_dissociated(new_S):
        return new_S
    return None

def construct():
    """
    Find a dissociated reciprocal set from [1, 22] with maximum cardinality.
    Uses hardcoded candidates to quickly get a size 11 set, then uses
    local search (add + swap) to maximize cardinality.
    """
    start_time = time.time()
    time_limit = 55  # Safety margin before 90s kill
    
    all_elements = list(range(1, 23))
    
    # Known good candidates to try first (fast path to size 11)
    # These are sets of size 11 that are likely dissociated.
    candidates = [
        [1, 2, 4, 8, 16, 17, 18, 19, 20, 21, 22],
        [1, 2, 4, 8, 16, 22, 21, 20, 19, 18, 17],
        [1, 2, 4, 8, 16, 19, 20, 21, 22, 18, 17],
        [1, 2, 4, 8, 16, 22, 20, 18, 16, 14, 12],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 21, 22], # Another potential size 11
    ]
    
    best_S = []
    best_size = 0
    current_S = []
    
    # 1. Fast Path: Find a valid size 11 set
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        # Ensure unique elements in candidate (just in case)
        cand_unique = list(set(cand))
        if _is_dissociated(cand_unique):
            if len(cand_unique) > best_size:
                best_S = cand_unique[:]
                best_size = len(best_S)
                current_S = best_S[:]
    
    # If we didn't find a size 11 set, try a quick greedy from scratch
    if best_size < 11:
        # Simple greedy to get a baseline
        S = []
        for x in all_elements:
            if time.time() - start_time > time_limit:
                break
            new_S = _try_add(S, x)
            if new_S is not None:
                S = new_S
        if len(S) > best_size:
            best_S = S[:]
            best_size = len(best_S)
            current_S = best_S[:]
            
    # If we still have nothing, start with a single element
    if not current_S:
        current_S = [1]
        best_S = [1]
        best_size = 1

    # 2. Local Search: Maximize cardinality
    # Strategy: 
    # - Try to add any remaining element.
    # - If stuck, perform random swaps to change the configuration.
    # - Accept swaps that maintain validity (even if size doesn't change immediately), 
    #   hoping to unlock future additions.
    
    no_improve_count = 0
    max_no_improve = 2000 # Reset if stuck too long
    
    while time.time() - start_time < time_limit:
        # Update best if current is better
        if len(current_S) > len(best_S):
            best_S = current_S[:]
            best_size = len(best_S)
            no_improve_count = 0
            
        added = False
        in_set = set(current_S)
        remaining = [x for x in all_elements if x not in in_set]
        
        # Try to add an element
        if remaining:
            # Shuffle to avoid bias towards specific elements
            random.shuffle(remaining)
            for cand in remaining:
                if time.time() - start_time > time_limit:
                    break
                new_S = _try_add(current_S, cand)
                if new_S is not None:
                    current_S = new_S
                    added = True
                    no_improve_count = 0
                    break # Restart loop with new set
        
        if not added:
            # Try swaps to escape local optima
            swapped = False
            if remaining and current_S:
                # Try a few random swaps
                # We limit the number of swap attempts per "stuck" phase to avoid wasting time
                # on a dead-end configuration
                max_swap_attempts = 50
                for _ in range(max_swap_attempts):
                    if time.time() - start_time > time_limit:
                        break
                    if not remaining or not current_S:
                        break
                    
                    out_cand = random.choice(remaining)
                    in_cand = random.choice(current_S)
                    
                    if out_cand == in_cand:
                        continue
                    
                    # Create new set
                    new_S = [x for x in current_S if x != in_cand] + [out_cand]
                    
                    if _is_dissociated(new_S):
                        current_S = new_S
                        # Update remaining list
                        in_set = set(current_S)
                        remaining = [x for x in all_elements if x not in in_set]
                        swapped = True
                        break
            
            if not swapped:
                no_improve_count += 1
                # If stuck for a while, reset to best or random restart
                if no_improve_count > max_no_improve:
                    # Reset to best to try different additions from a known good state
                    current_S = best_S[:]
                    no_improve_count = 0
                    
                    # Occasionally try a random restart of size len(best_S)
                    # to explore different parts of the search space
                    if random.random() < 0.1:
                        k = len(best_S)
                        # Sample k elements
                        random_sample = random.sample(all_elements, k)
                        if _is_dissociated(random_sample):
                            current_S = random_sample
                        else:
                            # Fallback to best
                            current_S = best_S[:]

    return sorted(best_S)

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
