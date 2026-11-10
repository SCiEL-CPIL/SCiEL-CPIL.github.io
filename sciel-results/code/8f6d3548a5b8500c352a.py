import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_cardinality_separation(elements):
    """Check if subset sums with different cardinalities are all distinct.
    Returns True if valid."""
    n = len(elements)
    # For each cardinality k, compute the set of achievable sums
    # sums[k] = set of all sums using exactly k elements
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    for e in elements:
        # iterate backwards to avoid using e multiple times
        for k in range(n - 1, -1, -1):
            for s in list(sums[k]):
                sums[k + 1].add(s + e)
    # Check disjointness across cardinalities
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums[i] & sums[j]:
                return False
    return True


def _check_cardinality_separation_fast(elements):
    """Faster check using the min/max gap property for sorted elements.
    For a sorted set, if min_sum(k) > max_sum(k-1) for all k, then
    all cardinality sums are separated. This is sufficient but not necessary."""
    elements = sorted(elements)
    n = len(elements)
    # min sum of k elements: sum of k smallest
    # max sum of k elements: sum of k largest
    prefix = [0] * (n + 1)
    suffix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + elements[i]
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + elements[i]
    for k in range(1, n + 1):
        min_sum_k = prefix[k]
        max_sum_k_minus_1 = suffix[1] if k - 1 == n else (elements[-1] + elements[-2] + ... ) if k > 1 else 0
        # max sum of k-1 elements = sum of k-1 largest
        if k - 1 == 0:
            max_sum_prev = 0
        else:
            max_sum_prev = 0
            for i in range(n - (k - 1), n):
                max_sum_prev += elements[i]
        if min_sum_k <= max_sum_prev:
            return False
    return True


def _check_separation_dp(elements):
    """Exact check: for each cardinality, compute achievable sums and verify disjointness."""
    n = len(elements)
    # sums_by_card[k] = set of achievable sums with exactly k elements
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    for e in elements:
        for k in range(n - 1, -1, -1):
            if sums_by_card[k]:
                for s in list(sums_by_card[k]):
                    sums_by_card[k + 1].add(s + e)
    # Check all pairs of cardinalities are disjoint
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_card[i] & sums_by_card[j]:
                return False
    return True


def _try_add_element(current_set, candidate, deadline):
    """Try to add candidate to current_set, check cardinality separation."""
    if time.time() > deadline:
        return False
    new_set = sorted(current_set + [candidate])
    return _check_separation_dp(new_set)


def construct():
    deadline = time.time() + 55  # safety margin
    
    # Start with the best arithmetic progression: {256-n+1, ..., 256}
    # Find max n for AP: need min_sum(k) > max_sum(k-1) for all k
    # For AP {b, b+1, ..., b+n-1}:
    # min_sum(k) = k*b + k*(k-1)//2
    # max_sum(k-1) = (k-1)*(b+n-1) - (k-1)*(k-2)//2
    # Gap = k*b + k*(k-1)//2 - (k-1)*(b+n-1) + (k-1)*(k-2)//2 > 0
    
    best_size = 0
    best_set = []
    
    # First, find the best AP
    for n in range(60, 0, -1):
        b = 256 - n + 1  # base of AP: {b, b+1, ..., 256}
        if b < 1:
            continue
        valid = True
        for k in range(1, n + 1):
            min_sum_k = k * b + k * (k - 1) // 2
            if k - 1 == 0:
                max_sum_prev = 0
            else:
                # sum of k-1 largest: (k-1)*(b+n-1) - (k-1)*(k-2)//2
                max_sum_prev = (k - 1) * (b + n - 1) - (k - 1) * (k - 2) // 2
            if min_sum_k <= max_sum_prev:
                valid = False
                break
        if valid:
            best_size = n
            best_set = list(range(b, b + n))
            break
    
    if not best_set:
        # Fallback: single element
        return [256]
    
    # Try to improve by replacing some elements with smaller ones to fit more
    # Strategy: start with the AP, try to add elements from the lower range
    current = sorted(best_set)
    used = set(current)
    
    # Try adding elements from 1 to 256 not in current, in order of most "beneficial"
    # An element is beneficial if it doesn't create collisions
    candidates = [x for x in range(1, 257) if x not in used]
    
    # Try adding candidates one by one (greedy)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in candidates[:]:
            if time.time() > deadline:
                break
            if cand in used:
                continue
            if _try_add_element(current, cand, deadline):
                current.append(cand)
                current.sort()
                used.add(cand)
                candidates.remove(cand)
                improved = True
                break
    
    # Verify final result
    if _check_separation_dp(current):
        return current
    else:
        # Fallback to the AP
        return best_set

# EVOLVE-BLOCK-END
