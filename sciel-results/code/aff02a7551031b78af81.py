import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 60.0  # Keep safety margin
    
    # Precompute LCMs or use math.lcm
    # Range 2 to 256
    # Objective: Max sum(1/x)
    
    def get_score(S):
        return sum(1.0/x for x in S)
        
    def is_compatible(a, b):
        return math.lcm(a, b) > 256
        
    # Initialize with Seed Solution
    # Seed: 2 and odd numbers > 128
    seed_set = [2]
    for x in range(129, 257, 2):
        seed_set.append(x)
    
    best_set = set(seed_set)
    best_score = get_score(best_set)
    
    # Local Search
    # Try to add numbers from 3 to 256 not in set
    # And try to swap
    
    candidates = list(range(2, 257))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order to escape local minima?
        # Or just iterate
        for x in candidates:
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            
            # Find conflicts
            conflicts = [y for y in best_set if not is_compatible(x, y)]
            
            # Calculate gain
            gain = 1.0/x
            loss = sum(1.0/y for y in conflicts)
            
            if gain > loss:
                # Add x, remove conflicts
                best_set.remove(x) if x in best_set else None # x not in set
                for y in conflicts:
                    best_set.remove(y)
                best_set.add(x)
                best_score += gain - loss
                improved = True
    
    # Try a different initialization: All Large
    large_set = set(range(129, 257))
    large_score = get_score(large_set)
    if large_score > best_score:
        best_set = large_set
        best_score = large_score
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in candidates:
                if x in best_set:
                    continue
                if time.time() > deadline:
                    break
                conflicts = [y for y in best_set if not is_compatible(x, y)]
                gain = 1.0/x
                loss = sum(1.0/y for y in conflicts)
                if gain > loss:
                    for y in conflicts:
                        best_set.remove(y)
                    best_set.add(x)
                    best_score += gain - loss
                    improved = True

    # Try one more: Greedy from scratch with random shuffles
    for _ in range(10):
        if time.time() > deadline:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        current_set = set()
        for x in shuffled:
            if all(is_compatible(x, y) for y in current_set):
                current_set.add(x)
        if get_score(current_set) > best_score:
            best_set = current_set
            best_score = get_score(best_set)

    return list(best_set)

# EVOLVE-BLOCK-END
