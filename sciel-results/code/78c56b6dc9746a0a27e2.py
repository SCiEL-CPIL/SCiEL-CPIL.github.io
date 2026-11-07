import time
from collections import defaultdict

def _sums_of_size_k(sorted_xs, k):
    """Yield all sums of exactly k elements from sorted_xs (distinct elements)."""
    n = len(sorted_xs)
    if k == 0:
        yield 0
        return
    if k > n:
        return
    if k == 1:
        for x in sorted_xs:
            yield x
        return
    if k == n:
        yield sum(sorted_xs)
        return
    # Use combinations for small k
    from itertools import combinations
    for combo in combinations(sorted_xs, k):
        yield sum(combo)

def _all_subset_sums_by_cardinality(xs):
    """Return list of sets, where index j is the set of all sums of j-element subsets."""
    n = len(xs)
    sorted_xs = sorted(xs)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for k in range(1, n + 1):
        for s in _sums_of_size_k(sorted_xs, k):
            sums_by_k[k].add(s)
    return sums_by_k

def _is_valid(xs):
    """Check that sums of different-cardinality subsets are disjoint."""
    sums_by_k = _all_subset_sums_by_cardinality(xs)
    n = len(xs)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_k[i] & sums_by_k[j]:
                return False
    return True

def _try_insertion(xs, candidate):
    """Try to add candidate to xs while maintaining validity. Returns new list or None."""
    new_xs = sorted(xs + [candidate])
    if _is_valid(new_xs):
        return new_xs
    return None

def _grow_greedy(initial_xs, pool, deadline):
    """Greedily add elements from pool to maintain validity."""
    current = list(initial_xs)
    for c in pool:
        if time.time() > deadline:
            break
        if c in current:
            continue
        new_xs = _try_insertion(current, [c])
        if new_xs is not None:
            current = new_xs
    return current

def _polish(xs, deadline):
    """Try to improve the set by swapping elements to allow more insertions."""
    current = list(xs)
    n = len(current)
    # Try removing one element and trying to add two
    for i in range(n):
        if time.time() > deadline:
            break
        without_i = current[:i] + current[i+1:]
        # Try adding various candidates
        for c in range(1, 257):
            if time.time() > deadline:
                break
            if c in without_i:
                continue
            new_xs = _try_insertion(without_i, [c])
            if new_xs is not None:
                # Try to add another
                for c2 in range(1, 257):
                    if time.time() > deadline:
                        break
                    if c2 in new_xs or c2 == c:
                        continue
                    new_xs2 = _try_insertion(new_xs, [c2])
                    if new_xs2 is not None and len(new_xs2) > len(current):
                        current = new_xs2
                        break
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second budget with 5s safety margin
    
    # Known best: 31 consecutive integers
    baseline = list(range(226, 257))
    best = baseline
    
    # Try to beat the baseline using greedy high-to-low
    # Start with the largest elements and greedily add smaller ones
    current = [256, 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241]
    
    # Greedily try to add elements from high to low
    for c in range(240, 0, -1):
        if time.time() > deadline:
            break
        if c in current:
            continue
        new_current = _try_insertion(current, [c])
        if new_current is not None:
            current = new_current
    
    if len(current) > len(best):
        best = current
    
    # Try polish
    polished = _polish(best, deadline)
    if len(polished) > len(best):
        best = polished
    
    # Verify the best result
    if not _is_valid(best):
        best = baseline
    
    # Ensure we return a list of Python ints
    return list(best)
