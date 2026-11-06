import time
import random
from itertools import combinations

def is_valid(S):
    """Check if set S satisfies the condition."""
    n = len(S)
    if n < 4:
        return True
    S_list = list(S)
    # Precompute all triple sums? No, depends on which element is excluded.
    # For each x in S, check triples from S \ {x}
    for i, x in enumerate(S_list):
        others = S_list[:i] + S_list[i+1:]
        # If others < 3, no triples, valid
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            if sum(t) % x == 0:
                return False
    return True

def get_conflicts(S):
    """Return list of conflicts (x, a, b, c) where x divides a+b+c."""
    conflicts = []
    S_list = list(S)
    n = len(S_list)
    for i, x in enumerate(S_list):
        others = S_list[:i] + S_list[i+1:]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            if sum(t) % x == 0:
                conflicts.append((x, t[0], t[1], t[2]))
    return conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_set = []
    
    # Strategy 1: Greedy from large numbers (similar to seed but optimized check)
    # Strategy 2: Local Search starting from [147..192]
    
    # Initial candidate pool: Large numbers are safer
    # We will run multiple restarts of local search
    
    while time.time() < deadline:
        # Initialize with a promising set
        # Start with [147, 192]
        current = list(range(147, 193))
        
        # Randomize start slightly to explore different basins
        if random.random() < 0.5:
            # Try a different range
            start = random.randint(140, 150)
            current = list(range(start, 193))
            if len(current) > 48:
                current = current[-48:]
        
        # Phase 1: Repair to valid state
        # Limit repair steps to avoid getting stuck
        repair_steps = 0
        max_repair_steps = 100
        
        while not is_valid(current) and repair_steps < max_repair_steps and time.time() < deadline - 10:
            conflicts = get_conflicts(current)
            if not conflicts:
                break
            # Pick a random conflict
            x, a, b, c = random.choice(conflicts)
            # Try removing one of the involved elements
            candidates_to_remove = [x, a, b, c]
            random.shuffle(candidates_to_remove)
            
            removed = False
            for r in candidates_to_remove:
                test = [v for v in current if v != r]
                if is_valid(test):
                    current = test
                    removed = True
                    break
            
            if not removed:
                # Fallback: remove x
                current = [v for v in current if v != x]
            
            repair_steps += 1
        
        if not is_valid(current):
            continue
        
        # Phase 2: Expand to size 48
        # Try adding elements
        while len(current) < 48 and time.time() < deadline - 10:
            candidates = [v for v in range(1, 193) if v not in current]
            if not candidates:
                break
            
            # Sort candidates by "safety" (larger is safer)
            # But randomize to avoid bias
            random.shuffle(candidates)
            
            added = False
            for y in candidates:
                test = current + [y]
                if is_valid(test):
                    current = test
                    added = True
                    break
            
            if not added:
                # Try swapping
                swapped = False
                # Try swapping one element of current with one from candidates
                # Limit swaps to save time
                for i in range(len(current)):
                    if swapped:
                        break
                    x = current[i]
                    for y in candidates:
                        test = current[:i] + [y] + current[i+1:]
                        if is_valid(test):
                            current = test
                            swapped = True
                            break
                
                if not swapped:
                    break
        
        if len(current) > len(best_set):
            best_set = current
    
    return best_set
