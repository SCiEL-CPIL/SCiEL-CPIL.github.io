"""Combined program: fast greedy + two-phase local search with interval-based selection."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

N = 192
K = 3
MAX_ELEM = 48

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints.
    Uses efficient checks: only new triples involving x need verification."""
    m = len(selected)
    
    # Check: does x divide sum of any 3 from selected?
    if m >= 3:
        sel_arr = np.array(selected)
        # Generate all triples sums efficiently
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    s = selected[i] + selected[j] + selected[k]
                    if s % x == 0:
                        return False
    
    # Check: for each a in selected, does a divide sum of any 3 from (selected ∪ {x}) \ {a}?
    # Only need to check triples that include x: x + 2 from selected\{a}
    for i, a in enumerate(selected):
        others_no_a = [selected[j] for j in range(m) if j != i]
        if len(others_no_a) >= 2:
            for j in range(len(others_no_a)):
                for k in range(j + 1, len(others_no_a)):
                    s = others_no_a[j] + others_no_a[k] + x
                    if s % a == 0:
                        return False
    return True

def _greedy_build(order, max_elem=MAX_ELEM):
    """Greedy build following given order."""
    selected = []
    for x in order:
        if len(selected) >= max_elem:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit, deadline=None):
    """Two-phase local search: first try additions, then swaps."""
    start = time.time()
    best = list(selected)
    
    def _time_ok():
        if deadline is not None:
            return time.time() < deadline
        return time.time() - start < time_limit
    
    while _time_ok():
        improved = False
        candidates = [x for x in range(N, 0, -1) if x not in best]
        
        # Phase 1: Try pure additions (cheaper, try larger first)
        for x in candidates:
            if not _time_ok():
                break
            if _can_add(x, best):
                best.append(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps (remove one, add one)
        # Try removing each element and see if we can add a better one
        for i in range(len(best)):
            if not _time_ok():
                break
            removed = best[i]
            remaining = best[:i] + best[i+1:]
            
            # Try adding candidates in descending order (larger = harder to divide)
            for x in candidates:
                if x == removed:
                    continue
                if _can_add(x, remaining):
                    best = remaining + [x]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _two_swap_local_search(selected, time_limit, deadline=None):
    """Enhanced local search with two-swap moves for escaping local optima."""
    start = time.time()
    best = list(selected)
    
    def _time_ok():
        if deadline is not None:
            return time.time() < deadline
        return time.time() - start < time_limit
    
    while _time_ok():
        improved = False
        candidates = [x for x in range(N, 0, -1) if x not in best]
        
        # Try single additions
        for x in candidates:
            if not _time_ok():
                break
            if _can_add(x, best):
                best.append(x)
                improved = True
                break
        if improved:
            continue
        
        # Try single swaps
        for i in range(len(best)):
            if not _time_ok():
                break
            removed = best[i]
            remaining = best[:i] + best[i+1:]
            for x in candidates:
                if x == removed:
                    continue
                if _can_add(x, remaining):
                    best = remaining + [x]
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try two-swap: remove two elements, add two new ones
        # This is expensive, so limit attempts
        if len(best) >= 2:
            for i in range(len(best)):
                if not _time_ok():
                    break
                for j in range(i + 1, len(best)):
                    if not _time_ok():
                        break
                    removed1, removed2 = best[i], best[j]
                    remaining = [v for k, v in enumerate(best) if k != i and k != j]
                    
                    # Try adding two new elements
                    for x in candidates[:20]:  # Limit to top 20 candidates for speed
                        if x == removed1 or x == removed2:
                            continue
                        if _can_add(x, remaining):
                            temp1 = remaining + [x]
                            for y in candidates[:20]:
                                if y == removed1 or y == removed2 or y == x:
                                    continue
                                if _can_add(y, temp1):
                                    best = temp1 + [y]
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 82.0  # 8s safety margin
    
    # Strategy 1: Greedy from large to small
    best = _greedy_build(list(range(N, 0, -1)))
    
    # Strategy 2: Greedy from small to large
    alt = _greedy_build(list(range(1, N + 1)))
    if len(alt) > len(best):
        best = alt
    
    # Strategy 3: Interval-based greedy with various hi_start values
    for hi_start in [192, 185, 175, 165, 155, 145, 135, 125, 115, 105]:
        if time.time() > deadline - 15:
            break
        order = list(range(hi_start, N + 1)) + list(range(hi_start - 1, 0, -1))
        alt = _greedy_build(order)
        if len(alt) > len(best):
            best = alt
    
    # Strategy 4: Randomized greedy with multiple seeds
    for seed in range(30):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        alt = _greedy_build(candidates)
        if len(alt) > len(best):
            best = alt
    
    # Local search on best result
    if time.time() < deadline - 5:
        remaining = min(deadline - time.time() - 2, 30.0)
        improved = _local_search(best, remaining, deadline=deadline)
        if len(improved) > len(best):
            best = improved
    
    # More random restarts with local search
    for seed in range(30, 80):
        if time.time() > deadline - 3:
            break
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        alt = _greedy_build(candidates)
        if len(alt) >= len(best) - 1:
            remaining = min(deadline - time.time() - 1, 5.0)
            if remaining > 0.5:
                alt = _local_search(alt, remaining, deadline=deadline)
                if len(alt) > len(best):
                    best = alt
    
    # Final polish with two-swap local search if time permits
    if time.time() < deadline - 2:
        remaining = min(deadline - time.time() - 1, 10.0)
        if remaining > 1.0:
            improved = _two_swap_local_search(best, remaining, deadline=deadline)
            if len(improved) > len(best):
                best = improved
    
    return best

# EVOLVE-BLOCK-END
