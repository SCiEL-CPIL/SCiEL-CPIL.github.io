import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def is_valid_float(S):
        if not S:
            return True
        sums = [0.0]
        for k in S:
            inv_k = 1.0 / k
            new_sums = [s + inv_k for s in sums]
            sums += new_sums
        sums.sort()
        for i in range(1, len(sums)):
            if sums[i] == sums[i-1]:
                return False
        return True

    candidates = list(range(1, 23))
    best_set = set()
    
    # Try to find a base set of size 16-22
    # We prioritize larger sizes first
    found_base = False
    for size in range(22, 9, -1):
        if time.time() > deadline:
            break
        # Try random samples
        for _ in range(200):
            if time.time() > deadline:
                break
            S = set(random.sample(candidates, size))
            if is_valid_float(S):
                best_set = S
                found_base = True
                break
        if found_base:
            break
            
    if not best_set:
        # Fallback to a known valid set (size 9)
        best_set = {1, 2, 3, 5, 7, 11, 13, 17, 19}
        
    # Try to grow best_set
    # If we found a large set, we might not need to grow much.
    # If we found a small set, we grow.
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try adding an element
        # Shuffle candidates to avoid bias towards small numbers
        random_candidates = candidates[:]
        random.shuffle(random_candidates)
        
        for k in random_candidates:
            if k in best_set:
                continue
            new_set = best_set | {k}
            if is_valid_float(new_set):
                best_set = new_set
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try swapping an element
        # Try to replace a "bad" element with a "good" one
        # We don't know which is bad, so try all swaps
        swapped = False
        for k_in in list(best_set):
            for k_out in candidates:
                if k_out in best_set:
                    continue
                new_set = (best_set - {k_in}) | {k_out}
                if is_valid_float(new_set):
                    best_set = new_set
                    swapped = True
                    break
            if swapped:
                break
        
        if not swapped:
            # Stuck. Try a random restart?
            # Or just break if we are at a high size.
            # Given the time limit, breaking is safer to return a valid result.
            # However, if we are stuck at size 16, we might want to try a different local search.
            # But simple hill climbing is often enough for this small N.
            break
            
    return list(best_set)
