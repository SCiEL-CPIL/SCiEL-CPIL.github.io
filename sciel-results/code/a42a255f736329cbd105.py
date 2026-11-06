"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def _valid_subset(S, target=200):
    """Check that no subset of S sums to target. S is a set of ints."""
    reachable = {0}
    for x in S:
        new = {s + x for s in reachable if s + x <= target}
        if target in new:
            return False
        reachable |= new
    return True

def _greedy_construct(order, target=200):
    """Greedy: include x if it doesn't create target in reachable sums."""
    selected = []
    reachable = {0}
    for x in order:
        new = {s + x for s in reachable if s + x <= target}
        if target not in new:
            selected.append(x)
            reachable |= new
    return selected

def construct():
    deadline = time.time() + 85  # safety margin before 90s
    best = []
    
    # Strategy 1: seed order (100 down to 1)
    best = _greedy_construct(list(range(100, 0, -1)))
    
    # Strategy 2: ascending
    cand = _greedy_construct(list(range(1, 101)))
    if len(cand) > len(best):
        best = cand
    
    # Strategy 3: multiple randomized greedy passes
    base = list(range(1, 101))
    for _ in range(300):
        if time.time() > deadline:
            break
        random.shuffle(base)
        cand = _greedy_construct(base)
        if len(cand) > len(best):
            best = cand
    
    # Strategy 4: local improvement - try to add back excluded elements
    best_set = set(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, 101):
            if x not in best_set:
                trial = best_set | {x}
                if _valid_subset(trial, 200):
                    best_set = trial
                    improved = True
                    break  # restart scan after adding
    
    best = list(best_set)
    
    # Final: ensure nonempty
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
