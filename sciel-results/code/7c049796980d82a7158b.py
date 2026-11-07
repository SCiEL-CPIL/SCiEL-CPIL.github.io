"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 256, 'maximum_size': 60}

def _check_valid(s):
    """Check if set s satisfies cardinality-separated subset sum property."""
    n = len(s)
    # reachable[k] = set of sums achievable with exactly k elements
    reachable = [set() for _ in range(n + 1)]
    reachable[0].add(0)
    for x in s:
        # Update in reverse to avoid using x multiple times
        for k in range(n, 0, -1):
            for sm in list(reachable[k - 1]):
                reachable[k].add(sm + x)
    # Check cross-cardinality: no sum appears in two different cardinality sets
    seen = {}
    for k in range(n + 1):
        for sm in reachable[k]:
            if sm in seen:
                return False
            seen[sm] = k
    return True

def _greedy_build(max_val=256, target_size=20, deadline=None):
    """Greedy: add elements one by one, preferring larger values first."""
    current = []
    used = set()
    candidates = list(range(max_val, 0, -1))  # try larger first
    random.shuffle(candidates)
    
    for cand in candidates:
        if len(current) >= target_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        if cand in used:
            continue
        test = current + [cand]
        test.sort()
        if _check_valid(test):
            current = test
            used.add(cand)
    return current

def construct():
    start = time.time()
    deadline = start + 50  # safety margin
    
    best = []
    
    # Try multiple random seeds and strategies
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        result = _greedy_build(max_val=256, target_size=60, deadline=deadline)
        if len(result) > len(best):
            best = result
            if len(best) >= 20:
                # Try to extend with smaller values
                for extra in range(1, 257):
                    if time.time() > deadline:
                        break
                    if extra not in best:
                        test = best + [extra]
                        test.sort()
                        if _check_valid(test):
                            best = test
    
    # Final validation
    if best and _check_valid(best):
        return list(best)
    return [1, 2, 4, 8, 16, 32, 64, 128]  # fallback: powers of 2
# EVOLVE-BLOCK-END
