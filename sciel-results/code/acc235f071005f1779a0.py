import random
import time

def is_valid(S):
    if not S:
        return True
    
    # Generate all non-empty subset sums
    sums = set()
    current = {0}
    for x in S:
        new_sums = set()
        for s in current:
            new_sums.add(s + x)
        current.update(new_sums)
    
    # Remove 0
    if 0 in current:
        current.remove(0)
    
    sums = current
    if not sums:
        return True
        
    max_sum = max(sums)
    
    # Check divisibility
    # Optimization: If a is small, many multiples to check.
    # If a is large, few.
    for a in sums:
        multiple = 2 * a
        while multiple <= max_sum:
            if multiple in sums:
                return False
            multiple += a
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    best_set = []
    best_size = 0
    
    # Try a few different strategies
    # 1. Greedy descending
    # 2. Random restarts
    
    # Strategy 1: Descending order
    candidates = list(range(256, 0, -1))
    current_set = []
    for x in candidates:
        trial = current_set + [x]
        if is_valid(trial):
            current_set = trial
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
                
    if time.time() > deadline:
        return best_set

    # Strategy 2: Random restarts
    while time.time() < deadline:
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        current_set = []
        for x in candidates:
            if time.time() > deadline:
                break
            trial = current_set + [x]
            if is_valid(trial):
                current_set = trial
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                    
    return best_set
