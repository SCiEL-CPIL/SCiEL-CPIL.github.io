"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    n = 128
    
    def check_validity(S_set):
        """Check if S_set has any forbidden triple (a,b,c distinct, all pair sums in S_set)."""
        elems = sorted(S_set)
        k = len(elems)
        for i in range(k):
            a = elems[i]
            for j in range(i+1, k):
                b = elems[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for k_idx in range(j+1, k):
                    c = elems[k_idx]
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True
    
    def count_forbidden(S_set):
        """Count forbidden triples for use as a penalty/objective."""
        elems = sorted(S_set)
        k = len(elems)
        count = 0
        for i in range(k):
            a = elems[i]
            for j in range(i+1, k):
                b = elems[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for k_idx in range(j+1, k):
                    c = elems[k_idx]
                    if (a + c) in S_set and (b + c) in S_set:
                        count += 1
        return count
    
    # Phase 1: Greedy construction
    S = set(range(64, n + 1))
    
    # Try adding elements from 1 to 63
    for x in range(63, 0, -1):
        if time.time() > deadline:
            break
        S_new = S | {x}
        if check_validity(S_new):
            S = S_new
    
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local search with swaps
    # Objective: maximize |S| while keeping validity. If invalid, minimize forbidden count.
    
    def objective(S_set):
        """Higher is better. Valid sets get high score, invalid get penalty."""
        if check_validity(S_set):
            return len(S_set)
        else:
            return len(S_set) - count_forbidden(S_set) * 10
    
    cur_S = set(S)
    cur_obj = objective(cur_S)
    
    # Hill climbing with random restarts
    rng = random.Random(42)
    all_elements = list(range(1, n + 1))
    
    while time.time() < deadline:
        # Try random moves: add one, remove one, or swap one
        move_type = rng.random()
        
        if move_type < 0.33:
            # Add a random element not in S
            candidates = [x for x in all_elements if x not in cur_S]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_S = cur_S | {x}
        elif move_type < 0.66:
            # Remove a random element from S
            if not cur_S:
                continue
            x = rng.choice(list(cur_S))
            new_S = cur_S - {x}
        else:
            # Swap: remove one, add one
            if not cur_S:
                continue
            x = rng.choice(list(cur_S))
            candidates = [y for y in all_elements if y not in cur_S]
            if not candidates:
                continue
            y = rng.choice(candidates)
            new_S = (cur_S - {x}) | {y}
        
        new_obj = objective(new_S)
        
        if new_obj >= cur_obj:
            cur_S = new_S
            cur_obj = new_obj
            if check_validity(cur_S) and len(cur_S) > best_size:
                best_S = set(cur_S)
                best_size = len(cur_S)
        else:
            # Accept with some probability to escape local optima (simulated annealing style)
            # Simple: accept if within a small penalty
            if cur_obj - new_obj <= 2:
                if rng.random() < 0.3:
                    cur_S = new_S
                    cur_obj = new_obj
    
    # Phase 3: Final greedy pass on best_S
    S_final = set(best_S)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_final:
            continue
        S_new = S_final | {x}
        if check_validity(S_new):
            S_final = S_new
    
    result = sorted(S_final)
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
