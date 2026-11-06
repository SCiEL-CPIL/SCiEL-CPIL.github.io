"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = []
    best_size = 0
    
    def greedy_forward(order):
        """Fast greedy: try elements in given order, add if valid."""
        chosen = []
        # Use a list for counts, size 2001 (max sum 1000+1000)
        counts = [0] * 2001
        
        for x in order:
            # Check 2*x
            if counts[2*x] >= 2:
                continue
            
            # Check x+y for all y in chosen
            valid = True
            for y in chosen:
                if counts[x + y] >= 2:
                    valid = False
                    break
            
            if valid:
                counts[2*x] += 1
                for y in chosen:
                    counts[x + y] += 1
                chosen.append(x)
        
        return chosen

    # Strategy 1: Forward greedy
    set1 = greedy_forward(range(1, 1001))
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1
    
    # Strategy 2: Backward greedy
    if time.time() < deadline:
        set2 = greedy_forward(range(1000, 0, -1))
        if len(set2) > best_size:
            best_size = len(set2)
            best_set = set2

    # Strategy 3: Randomized greedy
    random.seed(42)
    # Limit restarts to ensure time safety
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline - 10:
            break
        order = list(range(1, 1001))
        random.shuffle(order)
        set_r = greedy_forward(order)
        if len(set_r) > best_size:
            best_size = len(set_r)
            best_set = set_r

    # Strategy 4: Local improvement (fill-in)
    if time.time() < deadline - 5 and best_set:
        best_set = list(best_set)
        best_set.sort()
        # Rebuild counts
        counts = [0] * 2001
        for i in range(len(best_set)):
            a = best_set[i]
            for j in range(i, len(best_set)):
                b = best_set[j]
                counts[a + b] += 1
        
        # Try to add missing elements in increasing order
        # This is a heuristic: smaller numbers are more "flexible" or "constraining"?
        # Actually, just try all missing.
        missing = [x for x in range(1, 1001) if x not in best_set]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # We iterate over a snapshot of missing to avoid modification issues during iteration
            # But we break on first add, so it's fine.
            for x in missing:
                if counts[2*x] >= 2:
                    continue
                valid = True
                for y in best_set:
                    if counts[x + y] >= 2:
                        valid = False
                        break
                if valid:
                    # Add x
                    counts[2*x] += 1
                    for y in best_set:
                        counts[x + y] += 1
                    best_set.append(x)
                    best_set.sort()
                    improved = True
                    break

    return best_set
# EVOLVE-BLOCK-END
