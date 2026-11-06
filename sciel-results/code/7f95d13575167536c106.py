import random
import time

def _is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    n = len(A)
    max_size = 0
    # Check all subsets is too slow for n=12? 2^12 = 4096. It's fine.
    # But we can be smarter.
    # Actually, 4096 checks * 12*13/2 ops is very fast.
    best = 0
    for mask in range(1, 1 << n):
        sub = [A[i] for i in range(n) if (mask >> i) & 1]
        if len(sub) <= best:
            continue
        if _is_sidon(sub):
            best = len(sub)
            if best == n:
                break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to be safe
    
    best_set = list(range(12))
    best_val = _max_sidon_size(best_set)
    
    # Try to find a set with max sidon size 3
    # We know [0..11] has size 4 or 5.
    
    # Random initialization
    random.seed(42)
    
    current_set = sorted(random.sample(range(129), 12))
    current_val = _max_sidon_size(current_set)
    
    if current_val < best_val:
        best_set = current_set
        best_val = current_val
    
    while time.time() < deadline:
        # Local search: pick a random element, replace with random other
        # Or, if current_val > 3, try to break a specific Sidon subset
        
        if current_val <= 3:
            # We found a good candidate. Try to verify if we can get 2? Unlikely.
            # Just keep searching to see if we can find another one or confirm stability.
            # Or just break if we are confident.
            # Let's continue to see if we can improve (unlikely to go below 3).
            pass
        
        # Strategy: If current_val > 3, find a maximal Sidon subset S in current_set.
        # Then try to replace one element in S with a random element not in current_set.
        
        if current_val > 3:
            # Find a Sidon subset of size current_val
            n = len(current_set)
            target_size = current_val
            found_sub = None
            # Iterate masks of size target_size
            # Optimization: just check random subsets of that size?
            # Or iterate all.
            for mask in range(1, 1 << n):
                if bin(mask).count('1') != target_size:
                    continue
                sub = [current_set[i] for i in range(n) if (mask >> i) & 1]
                if _is_sidon(sub):
                    found_sub = sub
                    break
            
            if found_sub:
                # Pick a random element from found_sub to replace
                idx_to_replace = random.choice([current_set.index(x) for x in found_sub])
                # Pick a new random element
                new_elem = random.choice(range(129))
                if new_elem not in current_set:
                    old_elem = current_set[idx_to_replace]
                    current_set[idx_to_replace] = new_elem
                    current_set.sort()
                    new_val = _max_sidon_size(current_set)
                    if new_val < current_val:
                        current_val = new_val
                        if current_val < best_val:
                            best_val = current_val
                            best_set = current_set[:]
                    elif new_val == current_val:
                        # Accept with some probability to escape local minima
                        if random.random() < 0.5:
                            current_val = new_val
                    else:
                        # Revert
                        current_set[idx_to_replace] = old_elem
                        current_set.sort()
                else:
                    # Try swapping with an element outside the subset
                    outside_idx = [i for i in range(n) if current_set[i] not in found_sub]
                    if outside_idx:
                        idx_out = random.choice(outside_idx)
                        idx_in = random.choice([current_set.index(x) for x in found_sub])
                        current_set[idx_out], current_set[idx_in] = current_set[idx_in], current_set[idx_out]
                        current_set.sort()
                        new_val = _max_sidon_size(current_set)
                        if new_val < current_val:
                            current_val = new_val
                            if current_val < best_val:
                                best_val = current_val
                                best_set = current_set[:]
                        elif new_val == current_val:
                            if random.random() < 0.5:
                                current_val = new_val
                            else:
                                # Revert
                                current_set[idx_out], current_set[idx_in] = current_set[idx_in], current_set[idx_out]
                                current_set.sort()
                        else:
                            # Revert
                            current_set[idx_out], current_set[idx_in] = current_set[idx_in], current_set[idx_out]
                            current_set.sort()
        
        # Occasionally restart
        if random.random() < 0.05:
            current_set = sorted(random.sample(range(129), 12))
            current_val = _max_sidon_size(current_set)
            if current_val < best_val:
                best_val = current_val
                best_set = current_set[:]
        
        if best_val == 3:
            # We can't do better than 3 (likely). 
            # We can break early to save time if we want, but let's run a bit more to be sure.
            # Actually, if we find 3, it's a huge improvement. 
            # Let's just return it if we've spent enough time or just return it immediately?
            # Returning immediately is risky if there's a bug in _max_sidon_size.
            # But _max_sidon_size is exhaustive, so it's correct.
            # Let's run for a bit more to ensure we didn't get a fluke (though exhaustive check is deterministic).
            # If best_val is 3, it's 3.
            # We can break to save time for other evaluations? No, this is the only evaluation.
            # Let's break to return quickly.
            # Wait, the prompt says "Keep an EXPLICIT deadline guard".
            # If I return early, it's fine.
            pass

    return best_set
