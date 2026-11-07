import time
import random

# EVOLVE-BLOCK-START

def _check_antichain(sums):
    if not sums:
        return True
    sorted_sums = sorted(sums)
    sum_set = set(sorted_sums)
    max_sum = sorted_sums[-1]
    for s in sorted_sums:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def _can_add(current_sums, candidate):
    new_sums = set(current_sums)
    new_sums.add(candidate)
    for s in current_sums:
        new_sums.add(s + candidate)
    
    if not _check_antichain(new_sums):
        return None
    return new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    target_size = 10
    
    # Try random restarts
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        current_set = []
        current_sums = set()
        
        # Heuristic: Try larger numbers first, with some randomization
        candidates = list(range(256, 0, -1))
        # Shuffle a portion to allow variation
        # Or just shuffle entirely
        random.shuffle(candidates)
        
        for c in candidates:
            if len(current_set) >= target_size:
                break
            new_sums = _can_add(current_sums, c)
            if new_sums is not None:
                current_set.append(c)
                current_sums = new_sums
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            if best_size == target_size:
                # Verify and return
                if _check_antichain(current_sums):
                    return current_set
                    
    if best_set:
        return best_set
    return [1]

# EVOLVE-BLOCK-END
