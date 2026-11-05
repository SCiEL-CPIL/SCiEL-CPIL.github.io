"""
Small inclusion-maximal Sidon sets in [1,256].

Strategy:
1. Generate candidate maximal Sidon sets via randomized greedy with various orderings.
2. Use local search (removal-based) to try to shrink each maximal set while maintaining
   maximality.
3. Return the smallest set found within the time budget.
"""

import time
import random
import sys

N = 256
DEADLINE = time.time() + 55  # 55s safety margin within 60s wall-clock


def _all_sums(s):
    """Return the set of all pair sums a+b with a<=b from the sorted list s."""
    sums = set()
    n = len(s)
    for i in range(n):
        sums.add(2 * s[i])
        for j in range(i + 1, n):
            sums.add(s[i] + s[j])
    return sums


def _is_sidon(s):
    """Check if s is a Sidon set (all pair sums a+b with a<=b are distinct)."""
    seen = set()
    n = len(s)
    for i in range(n):
        val = 2 * s[i]
        if val in seen:
            return False
        seen.add(val)
        for j in range(i + 1, n):
            val = s[i] + s[j]
            if val in seen:
                return False
            seen.add(val)
    return True


def _is_maximal(s, N):
    """Check if s is inclusion-maximal: adding any missing element from [1,N] breaks Sidon."""
    s_set = set(s)
    sums = _all_sums(s)
    for x in range(1, N + 1):
        if x in s_set:
            continue
        # Check if adding x would create a duplicate sum
        new_sums = set()
        for a in s:
            new_sums.add(a + x)
        new_sums.add(2 * x)
        if sums.intersection(new_sums):
            continue
        # If no collision, s is NOT maximal
        return False
    return True


def _greedy_construct(order, N):
    """
    Greedy Sidon set construction following a given order.
    Returns (selected_list, sums_set).
    """
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added
        new_sums = [x + a for a in selected] + [2 * x]
        if len(set(new_sums)) == len(new_sums) and not sums.intersection(new_sums):
            selected.append(x)
            sums.update(new_sums)
    return selected, sums


def _try_remove(s, N):
    """
    Try to remove elements from s while maintaining maximality.
    Returns a list of maximal Sidon sets (possibly including s itself).
    """
    s_list = list(s)
    best = list(s_list)
    
    # Try removing each element one at a time, then try more removals
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        for i in range(len(best)):
            candidate = best[:i] + best[i+1:]
            if len(candidate) == 0:
                continue
            if _is_sidon(candidate) and _is_maximal(candidate, N):
                best = candidate
                changed = True
                break
    return best


def _local_search(s, N, max_iters=200):
    """
    Local search: try removing elements, and if stuck, try swap moves.
    Returns the best (smallest) maximal Sidon set found.
    """
    best = list(s)
    best_size = len(best)
    
    for _ in range(max_iters):
        if time.time() >= DEADLINE:
            break
        
        improved = False
        # Try single removals
        for i in range(len(best)):
            candidate = best[:i] + best[i+1:]
            if len(candidate) == 0:
                continue
            if _is_sidon(candidate) and _is_maximal(candidate, N):
                best = candidate
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one, add another missing element
        s_set = set(best)
        for i in range(len(best)):
            for x in range(1, N + 1):
                if x in s_set:
                    continue
                candidate = best[:i] + best[i+1:] + [x]
                candidate.sort()
                if _is_sidon(candidate) and _is_maximal(candidate, N):
                    if len(candidate) < best_size:
                        best = candidate
                        best_size = len(best)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return best


def _random_order(seed):
    """Generate a random permutation of [1, N] with a given seed."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return order


def _biased_order(seed, bias_low=True):
    """
    Generate an order with bias: either prefer small or large numbers.
    This can produce different maximal sets.
    """
    rng = random.Random(seed)
    base = list(range(1, N + 1))
    rng.shuffle(base)
    if bias_low:
        # Sort by random weight but with a pull toward small numbers
        weighted = [(x, rng.random() + x / (N + 1) * 0.5) for x in base]
    else:
        weighted = [(x, rng.random() + (N - x) / (N + 1) * 0.5) for x in base]
    weighted.sort(key=lambda t: t[1])
    return [x for x, _ in weighted]


def construct():
    global DEADLINE
    DEADLINE = time.time() + 55
    
    best_set = None
    best_size = N + 1
    
    # Phase 1: Generate diverse maximal Sidon sets via different orderings
    seeds_to_try = list(range(200))
    
    for seed in seeds_to_try:
        if time.time() >= DEADLINE:
            break
        
        # Use different ordering strategies
        if seed % 3 == 0:
            order = _random_order(seed)
        elif seed % 3 == 1:
            order = _biased_order(seed, bias_low=True)
        else:
            order = _biased_order(seed, bias_low=False)
        
        selected, sums = _greedy_construct(order, N)
        
        if not selected:
            continue
        
        # Verify it's Sidon and maximal
        if not _is_sidon(selected):
            continue
        if not _is_maximal(selected, N):
            # Try to make it maximal by adding elements
            # Actually, greedy should produce a maximal set if it processes all elements.
            # But let's verify. If not maximal, skip or fix.
            continue
        
        size = len(selected)
        if size < best_size:
            best_size = size
            best_set = list(selected)
        
        # Phase 2: Local search to shrink
        if size <= best_size:
            improved = _local_search(selected, N, max_iters=100)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
    
    # Fallback: deterministic ascending greedy
    if best_set is None:
        selected, sums = _greedy_construct(list(range(1, N + 1)), N)
        if _is_sidon(selected) and _is_maximal(selected, N):
            best_set = selected
    
    if best_set is None:
        # Last resort: just return [1] which is trivially Sidon but not maximal.
        # Actually [1] is not maximal. Let's use a small known maximal set.
        # Just return the ascending greedy result even if not verified.
        selected, sums = _greedy_construct(list(range(1, N + 1)), N)
        best_set = selected
    
    return best_set


# Quick self-test if run directly
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Size: {len(result)}, Time: {elapsed:.2f}s")
    print(f"Set: {result}")
    # Verify
    assert _is_sidon(result), "Not a Sidon set!"
    assert _is_maximal(result, N), "Not maximal!"
    print("Valid maximal Sidon set.")
