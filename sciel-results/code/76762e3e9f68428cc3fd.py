import time

def _check_validity(A):
    """Check if set A satisfies cardinality-separated subset sums."""
    n = len(A)
    if n == 0:
        return True
    # sums_by_k[k] = set of achievable sums using exactly k elements
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    for a in A:
        # Update in reverse to avoid using a twice
        for k in range(n - 1, -1, -1):
            for s in list(sums_by_k[k]):
                sums_by_k[k + 1].add(s + a)
    # Check no sum appears in two different cardinality sets
    seen = {}
    for k in range(n + 1):
        for s in sums_by_k[k]:
            if s in seen:
                return False
            seen[s] = k
    return True

def _check_validity_fast(A):
    """Faster check using the min/max separation heuristic first, then exact DP if needed."""
    n = len(A)
    if n == 0:
        return True
    A_sorted = sorted(A)
    # Quick necessary condition: for each k, min sum of (k+1) smallest > max sum of k largest
    # This is sufficient but not necessary. Use as fast filter.
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + A_sorted[i]
    # max sum of k largest = total - sum of (n-k) smallest = total - prefix[n-k]
    total = prefix[n]
    for k in range(n):
        min_sum_k1 = prefix[k + 1]  # sum of k+1 smallest
        max_sum_k = total - prefix[n - k]  # sum of k largest
        if min_sum_k1 <= max_sum_k:
            # Not necessarily invalid, but likely. Do exact check.
            return _check_validity(A)
    return True

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    # Greedy: try to build largest valid set from 1..256
    # Start with a known good seed and try to expand
    # Use powers of 2 as initial, then try to add more
    
    # Better: greedy from smallest, try to add each candidate
    A = []
    for candidate in range(1, 257):
        if time.time() - start_time > deadline:
            break
        new_A = A + [candidate]
        if _check_validity_fast(new_A):
            A = new_A
    
    # If we have time, try to improve by swapping
    if time.time() - start_time < deadline - 5:
        # Try to replace some elements to fit more
        # Actually, let's try a different greedy: prioritize keeping the set valid
        # while trying to maximize count. The above greedy already does this.
        # Let's try to see if we can do better with a different order.
        pass
    
    # Ensure we return Python ints in a list
    return [int(x) for x in A]
