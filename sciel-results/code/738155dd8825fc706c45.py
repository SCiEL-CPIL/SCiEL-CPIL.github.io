"""Rewritten: randomized greedy with exact cardinality-indexed subset-sum DP."""
import time
import random
from itertools import combinations

DEADLINE = 85  # safety margin before 90s kill

def _check_validity(elements):
    """Check if the set satisfies the cardinality-separated subset sum constraint."""
    n = len(elements)
    # sums_by_card[k] = set of achievable sums using exactly k elements
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for x in elements:
        # Update in reverse to avoid using x multiple times
        for k in range(len(sums_by_card) - 1, 0, -1):
            for s in list(sums_by_card[k - 1]):
                sums_by_card[k].add(s + x)
    
    # Check pairwise disjointness for different cardinalities
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_card[i] & sums_by_card[j]:
                return False
    return True

def _greedy_build(max_size, rng, start_set=None):
    """Greedy build: try to add elements one by one."""
    if start_set is None:
        elements = []
    else:
        elements = list(start_set)
    
    # Precompute valid candidates
    all_candidates = list(range(1, 257))
    rng.shuffle(all_candidates)
    
    for candidate in all_candidates:
        if candidate in elements:
            continue
        if len(elements) >= max_size:
            break
        test_set = elements + [candidate]
        if _check_validity(test_set):
            elements.append(candidate)
    
    return elements

def construct():
    start_time = time.time()
    best_set = []
    
    # Try multiple random restarts with different strategies
    rng = random.Random(42)
    
    # Strategy 1: Pure random greedy
    for attempt in range(50):
        if time.time() - start_time > DEADLINE:
            break
        s = _greedy_build(60, rng)
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 2: Start with powers of 2, then greedily add
    base = [1, 2, 4, 8, 16, 32, 64, 128]
    for attempt in range(30):
        if time.time() - start_time > DEADLINE:
            break
        rng2 = random.Random(100 + attempt)
        s = _greedy_build(60, rng2, start_set=base)
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 3: Start with geometric progression
    for ratio in [3, 4, 5]:
        if time.time() - start_time > DEADLINE:
            break
        geo = []
        v = 1
        while v <= 256:
            geo.append(v)
            v *= ratio
        geo = geo[:12]  # cap
        for attempt in range(10):
            if time.time() - start_time > DEADLINE:
                break
            rng3 = random.Random(200 + ratio * 10 + attempt)
            s = _greedy_build(60, rng3, start_set=geo)
            if len(s) > len(best_set):
                best_set = s
    
    # Final: try to improve best_set with more random additions
    if best_set and time.time() - start_time < DEADLINE - 5:
        for attempt in range(20):
            if time.time() - start_time > DEADLINE:
                break
            rng4 = random.Random(500 + attempt)
            s = _greedy_build(60, rng4, start_set=best_set)
            if len(s) > len(best_set):
                best_set = s
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
