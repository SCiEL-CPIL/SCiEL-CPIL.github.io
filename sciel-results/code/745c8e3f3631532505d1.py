import random
import time

def _can_add(x, S, counts):
    """Check if x can be added to S without violating B2[2]."""
    for y in S:
        if y == x:
            continue
        s = x + y
        if counts.get(s, 0) >= 2:
            return False
    s = 2 * x
    if counts.get(s, 0) >= 2:
        return False
    return True

def _do_add(x, S, counts):
    """Add x to S and update counts."""
    S.add(x)
    for y in S:
        if y == x:
            continue
        s = x + y
        counts[s] = counts.get(s, 0) + 1
    s = 2 * x
    counts[s] = counts.get(s, 0) + 1

def _do_remove(x, S, counts):
    """Remove x from S and update counts."""
    S.remove(x)
    for y in S:
        s = x + y
        counts[s] = counts.get(s, 0) - 1
        if counts[s] == 0:
            del counts[s]
    s = 2 * x
    counts[s] = counts.get(s, 0) - 1
    if counts[s] == 0:
        del counts[s]

def _greedy_solve(order):
    chosen = []
    counts = {}
    for x in order:
        valid = True
        for y in chosen:
            if counts.get(x + y, 0) >= 2:
                valid = False
                break
        if valid:
            if counts.get(2 * x, 0) >= 2:
                valid = False
        
        if valid:
            chosen.append(x)
            for y in chosen[:-1]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            s = 2 * x
            counts[s] = counts.get(s, 0) + 1
    return chosen

def _local_search(S, time_limit=5):
    S = set(S)
    counts = {}
    S_list = sorted(S)
    for i in range(len(S_list)):
        x = S_list[i]
        for j in range(i+1, len(S_list)):
            y = S_list[j]
            counts[x+y] = counts.get(x+y, 0) + 1
        counts[2*x] = counts.get(2*x, 0) + 1
        
    start_time = time.time()
    while (time.time() - start_time) < time_limit:
        progress = False
        
        # 1. Try to add
        # Sample random candidates
        # Instead of shuffling all, just try random ones
        for _ in range(100):
            x = random.randint(1, 1000)
            if x in S:
                continue
            if _can_add(x, S, counts):
                _do_add(x, S, counts)
                progress = True
                break
        
        if progress:
            continue
            
        # 2. Try to swap
        swap_attempts = 50
        for _ in range(swap_attempts):
            if not S:
                break
            y = random.choice(list(S))
            x = random.randint(1, 1000)
            if x in S:
                continue
            
            _do_remove(y, S, counts)
            if _can_add(x, S, counts):
                _do_add(x, S, counts)
                progress = True
                break
            else:
                _do_add(y, S, counts)
        
        if not progress:
            # Stuck. No adds, no swaps.
            # We can break or kick.
            # In ILS, we handle kicking outside.
            # Here, we just break to return current local optimum.
            break
            
    return S

def construct():
    candidates = list(range(1, 1001))
    
    # Initial Greedy
    best_set = _greedy_solve(candidates)
    
    start_time = time.time()
    total_time_limit = 110
    
    while (time.time() - start_time) < total_time_limit:
        # Run Local Search on a copy of best_set
        S = set(best_set)
        S = _local_search(S, time_limit=5)
        
        if len(S) > len(best_set):
            best_set = list(S)
        else:
            # Perturb best_set
            if best_set:
                # Remove 1-3 random elements
                num_remove = random.randint(1, 3)
                to_remove = random.sample(best_set, min(num_remove, len(best_set)))
                for x in to_remove:
                    best_set.remove(x)
                
                # Run Local Search on perturbed set
                S = set(best_set)
                S = _local_search(S, time_limit=5)
                
                if len(S) > len(best_set):
                    best_set = list(S)
                    
    return best_set
