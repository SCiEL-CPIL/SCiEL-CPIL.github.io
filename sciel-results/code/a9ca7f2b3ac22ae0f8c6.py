"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128

def is_valid(S):
    """Check if set S has no three distinct a,b,c with a+b,a+c,b+c all in S."""
    S_set = set(S)
    elems = list(S_set)
    n = len(elems)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = elems[i], elems[j], elems[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def count_violations(S):
    """Count number of violating triples."""
    S_set = set(S)
    elems = list(S_set)
    n = len(elems)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = elems[i], elems[j], elems[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Start with upper half
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    # Try to add elements from lower half
    # Greedy: try adding each element from 1 to 64
    for x in range(1, 65):
        if time.time() > deadline - 10:
            break
        new_set = best_set | {x}
        if is_valid(new_set):
            best_set = new_set
            best_size = len(best_set)
    
    # Local search: try to add more elements
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for x in range(1, 129):
            if x in best_set:
                continue
            new_set = best_set | {x}
            if is_valid(new_set):
                best_set = new_set
                best_size = len(best_set)
                changed = True
    
    # Try removing elements to allow adding more
    # For each element in the set, try removing it and see if we can add more
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        improved = False
        for x in sorted(best_set):
            if time.time() > deadline - 5:
                break
            trial_set = best_set - {x}
            # Try to add elements
            added = 0
            for y in range(1, 129):
                if y in trial_set:
                    continue
                new_set = trial_set | {y}
                if is_valid(new_set):
                    trial_set = new_set
                    added += 1
            if len(trial_set) > len(best_set):
                best_set = trial_set
                best_size = len(best_set)
                improved = True
    
    return list(best_set)
# EVOLVE-BLOCK-END
