"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial heuristic: Upper tail
    best_set = set(range(171, 257))
    best_size = len(best_set)
    
    def is_valid_add(S, x):
        # Check if adding x to S keeps it valid
        # S is a set
        # 1. x acts as a (smallest)
        for b in S:
            if b > x:
                for c in S:
                    if c >= b and (b + c) % x == 0:
                        return False
        # 2. x acts as b (middle)
        for a in S:
            if a < x:
                for c in S:
                    if c >= x and (x + c) % a == 0:
                        return False
        # 3. x acts as c (largest)
        for a in S:
            for b in S:
                if a < b <= x and (b + x) % a == 0:
                        return False
        return True

    candidates = list(range(1, N + 1))
    
    # Local search loop
    while time.time() < deadline:
        current_set = set(best_set)
        
        # Try to improve by adding elements
        # We do a greedy pass with randomization
        # Shuffle candidates
        random.shuffle(candidates)
        for x in candidates:
            if x not in current_set:
                if is_valid_add(current_set, x):
                    current_set.add(x)
        
        # If we improved, update best
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            continue # Restart greedy from the new best
            
        # If no improvement, try a "kick"
        # Remove k random elements
        k = random.randint(2, 5)
        kick_set = set(random.sample(list(current_set), k))
        current_set -= kick_set
        
        # Greedily add back
        random.shuffle(candidates)
        for x in candidates:
            if x not in current_set:
                if is_valid_add(current_set, x):
                    current_set.add(x)
                    
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)

    return list(best_set)
# EVOLVE-BLOCK-END
