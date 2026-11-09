import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, leaving 10s margin
    
    max_val = 4096
    
    best_set = []
    best_size = 0
    
    # Helper to check if candidate x can be added to current_set
    # current_set: sorted list of ints
    # sum_set: set of all valid triple sums from current_set
    def can_add(x, current_set, sum_set):
        n = len(current_set)
        new_sums = set()
        
        # (x, x, x)
        s = 3 * x
        if s in sum_set or s in new_sums:
            return False, None
        new_sums.add(s)
        
        # (x, x, y) for each y in current_set
        for y in current_set:
            s = 2 * x + y
            if s in sum_set or s in new_sums:
                return False, None
            new_sums.add(s)
            
        # (x, y, y) for each y in current_set
        for y in current_set:
            s = x + 2 * y
            if s in sum_set or s in new_sums:
                return False, None
            new_sums.add(s)
            
        # (x, y, z) with y < z from current_set
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                s = x + y + z
                if s in sum_set or s in new_sums:
                    return False, None
                new_sums.add(s)
            
        return True, new_sums

    # Strategy: Use greedy approach with different orderings
    # Try multiple strategies:
    # 1. Greedy from large to small
    # 2. Greedy from small to large  
    # 3. Randomized greedy
    
    # First, try a deterministic greedy from large to small
    # This tends to work well for Sidon-like sets
    current_set = []
    sum_set = set()
    
    # Start with a good seed - pick numbers that are well-spaced
    # For B_3[1], we want numbers that grow superlinearly
    # Try starting with a geometric-like sequence
    
    # Strategy 1: Greedy from largest to smallest
    candidates_desc = list(range(max_val, 0, -1))
    for x in candidates_desc:
        ok, new_sums = can_add(x, current_set, sum_set)
        if ok:
            current_set.append(x)
            current_set.sort()
            sum_set.update(new_sums)
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]
    
    # Strategy 2: Try randomized approaches with different biases
    random.seed(42)
    
    for trial in range(200):
        if time.time() >= deadline:
            break
        
        # Create a biased random ordering
        # Bias towards larger numbers or use geometric spacing
        candidates = list(range(1, max_val + 1))
        
        # Apply a random bias: mix of uniform and size-biased
        if trial % 3 == 0:
            # Descending with some randomness
            random.shuffle(candidates)
            candidates.sort(reverse=True)
        elif trial % 3 == 1:
            # Size-biased: larger numbers more likely to be tried first
            weights = [c ** 2 for c in candidates]
            total = sum(weights)
            # Sample without replacement with size bias
            selected = []
            remaining = candidates[:]
            rem_weights = weights[:]
            for _ in range(len(candidates)):
                if not remaining:
                    break
                r = random.random() * sum(rem_weights)
                cumsum = 0
                for idx, w in enumerate(rem_weights):
                    cumsum += w
                    if r <= cumsum:
                        selected.append(remaining[idx])
                        remaining.pop(idx)
                        rem_weights.pop(idx)
                        break
            candidates = selected
        else:
            # Pure random
            random.shuffle(candidates)
        
        current_set = []
        sum_set = set()
        
        for x in candidates:
            if time.time() >= deadline - 5:
                break
            ok, new_sums = can_add(x, current_set, sum_set)
            if ok:
                current_set.append(x)
                current_set.sort()
                sum_set.update(new_sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 42:
                break
    
    return best_set
