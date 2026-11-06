"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 512

def _is_4ap_free(S_set):
    """Check if a set contains no 4-term arithmetic progression."""
    S = set(S_set)
    n = len(S)
    if n < 4:
        return True
    S_sorted = sorted(S)
    # For each pair (a, c) with a < c, check if b and d form a 4-AP
    # 4-AP: x, x+d, x+2d, x+3d
    # So we need x, x+d, x+2d, x+3d all in S
    # Iterate over possible starts and differences
    for x in S_sorted:
        # x is the first term
        # x+3d <= N => d <= (N-x)//3
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
    return True

def _greedy_construct(order):
    """Greedy construction given an ordering of candidates."""
    selected = set()
    for x in order:
        if x in selected:
            continue
        # Check if adding x creates a 4-AP
        # x can be the 1st, 2nd, 3rd, or 4th term of a 4-AP
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        ok = True
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x+d in selected) and (x+2*d in selected) and (x+3*d in selected):
                ok = False
                break
        if ok:
            # Case 2: x is 2nd term: x-d, x, x+d, x+2d
            for d in range(1, x//3 + 1):
                if (x-d in selected) and (x+d in selected) and (x+2*d in selected):
                    ok = False
                    break
        if ok:
            # Case 3: x is 3rd term: x-2d, x-d, x, x+d
            for d in range(1, (x-1)//2 + 1):
                if (x-2*d in selected) and (x-d in selected) and (x+d in selected):
                    ok = False
                    break
        if ok:
            # Case 4: x is 4th term: x-3d, x-2d, x-d, x
            for d in range(1, (x-1)//3 + 1):
                if (x-3*d in selected) and (x-2*d in selected) and (x-d in selected):
                    ok = False
                    break
        if ok:
            selected.add(x)
    return selected

def _local_search(S_set, time_limit=45.0):
    """Improve the set via local search (additions, removals, swaps)."""
    selected = set(S_set)
    best = set(selected)
    best_size = len(best)
    start_time = time.time()
    
    # Precompute for speed: use a boolean array
    in_set = np.zeros(N+1, dtype=bool)
    for x in selected:
        in_set[x] = True
    
    def can_add(x):
        if in_set[x]:
            return False
        # Check all 4 roles
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if in_set[x+d] and in_set[x+2*d] and in_set[x+3*d]:
                return False
        for d in range(1, x//3 + 1):
            if in_set[x-d] and in_set[x+d] and in_set[x+2*d]:
                return False
        for d in range(1, (x-1)//2 + 1):
            if in_set[x-2*d] and in_set[x-d] and in_set[x+d]:
                return False
        for d in range(1, (x-1)//3 + 1):
            if in_set[x-3*d] and in_set[x-2*d] and in_set[x-d]:
                return False
        return True
    
    def try_add(x):
        if can_add(x):
            in_set[x] = True
            selected.add(x)
            return True
        return False
    
    def remove(x):
        in_set[x] = False
        selected.discard(x)
    
    # Try adding elements
    progress = True
    while progress and (time.time() - start_time) < time_limit:
        progress = False
        for x in range(1, N+1):
            if (time.time() - start_time) > time_limit:
                break
            if x not in selected and can_add(x):
                in_set[x] = True
                selected.add(x)
                progress = True
                if len(selected) > best_size:
                    best = set(selected)
                    best_size = len(best)
    
    # Now try swaps: remove one, add two
    # This is more complex; let's do simple swap: remove one, add one (if it leads to net gain later)
    # Actually, let's try: for each element y in selected, try removing it and then greedily adding
    # any new elements that become possible.
    
    # Simpler approach: random restarts with different orderings
    # But first, let's try a few more passes of addition
    
    return best

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    # Start with a greedy construction
    # Use a natural order first
    initial_order = list(range(1, N+1))
    S = _greedy_construct(initial_order)
    best = set(S)
    best_size = len(best)
    
    # Try local search to add more elements
    S_improved = _local_search(S, time_limit=min(30.0, deadline - (time.time() - start_time)))
    if len(S_improved) > best_size:
        best = set(S_improved)
        best_size = len(best)
    
    # Try different random orderings for greedy
    random.seed(42)
    num_trials = 0
    while (time.time() - start_time) < deadline:
        num_trials += 1
        order = list(range(1, N+1))
        random.shuffle(order)
        S_trial = _greedy_construct(order)
        if len(S_trial) > best_size:
            best = set(S_trial)
            best_size = len(best)
            # Try to improve this better solution
            S_improved = _local_search(S_trial, time_limit=min(10.0, deadline - (time.time() - start_time)))
            if len(S_improved) > best_size:
                best = set(S_improved)
                best_size = len(best)
    
    # Final local search on the best solution
    remaining = deadline - (time.time() - start_time)
    if remaining > 5.0:
        S_final = _local_search(best, time_limit=remaining - 2.0)
        if len(S_final) > best_size:
            best = set(S_final)
            best_size = len(best)
    
    return sorted(best)
# EVOLVE-BLOCK-END
