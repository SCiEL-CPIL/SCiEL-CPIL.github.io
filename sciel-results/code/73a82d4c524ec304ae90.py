import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_valid(set_ints):
    """Check that two subsets of different cardinalities never share a sum."""
    n = len(set_ints)
    # sums[k] = set of achievable sums with exactly k elements
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    for x in set_ints:
        # update backwards to avoid reusing x
        for k in range(n, 0, -1):
            for s in list(sums[k - 1]):
                sums[k].add(s + x)
    # check cross-cardinality collisions
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums[i] & sums[j]:
                return False
    return True

def _can_add(set_ints, x):
    """Check if adding x to set_ints preserves the property."""
    new_set = set_ints + [x]
    n = len(new_set)
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    for v in new_set:
        for k in range(n, 0, -1):
            for s in list(sums[k - 1]):
                sums[k].add(s + v)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums[i] & sums[j]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s

    # Greedy construction: pick smallest available integer that preserves validity
    # Start with a strong seed that is known to work well
    # Use a multiplicative gap approach for initial seed, then fill gaps greedily
    
    # Initial seed: geometric sequence gives guaranteed separation
    # If a_i >= 1 + sum of all previous, then subset sums are unique per cardinality
    # But that's too restrictive. Let's use a moderate gap.
    
    # Start with a small valid set and grow
    current = []
    
    # Try to build from scratch greedily, smallest-first
    # For speed, use a smarter initialization
    # A set like {1, 3, 7, 15, ...} (2^k - 1) works perfectly but is too sparse
    # Let's try a denser approach
    
    # Strategy: greedy smallest-first with DP validation
    # This is O(256 * n * max_sum) per candidate which is manageable
    
    # Use a faster validation: for each cardinality k, track min and max sums
    # and check if ranges overlap between different cardinalities
    # Actually we need exact sets, but for a quick filter we can use ranges
    
    def quick_reject(set_ints, x):
        """Quick range-based rejection: if sum ranges for different k overlap, reject."""
        new_set = set_ints + [x]
        n = len(new_set)
        # For cardinality k, min sum = sum of k smallest, max sum = sum of k largest
        sorted_new = sorted(new_set)
        prefix = [0]
        for v in sorted_new:
            prefix.append(prefix[-1] + v)
        # min sum for k elements = prefix[k]
        # max sum for k elements = prefix[n] - prefix[n-k]
        for k in range(n + 1):
            min_k = prefix[k]
            max_k = prefix[n] - prefix[n - k]
            for j in range(n + 1):
                if j == k:
                    continue
                min_j = prefix[j]
                max_j = prefix[n] - prefix[n - j]
                if min_k <= max_j and min_j <= max_k:
                    # ranges overlap, but this is a necessary not sufficient condition
                    # don't reject on this alone, it's too conservative
                    pass
        return False  # quick reject is too conservative, skip it
    
    # Full DP-based greedy: try each candidate from 1 to 256
    current = []
    used = set()
    
    for target_size in range(1, 61):
        if time.time() > deadline:
            break
        found = False
        for x in range(1, 257):
            if x in used:
                continue
            if _can_add(current, x):
                current.append(x)
                used.add(x)
                found = True
                break
        if not found:
            break
    
    # Try to improve: swap elements to allow denser packing
    # Actually, let's try a different strategy: random restarts with different orderings
    best = list(current)
    
    if time.time() < deadline - 5:
        # Try multiple random orderings
        import random
        rng = random.Random(42)
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
            # Randomize the order of trying candidates
            candidates = list(range(1, 257))
            rng.shuffle(candidates)
            # But bias towards smaller values for density
            # Actually, smaller values are better for fitting more in 256
            # Let's try: sorted with some random perturbation
            candidates = list(range(1, 257))
            # Perturb: for each position, swap with a random nearby position
            for i in range(len(candidates) - 1):
                j = min(i + rng.randint(0, 10), len(candidates) - 1)
                candidates[i], candidates[j] = candidates[j], candidates[i]
            
            cur = []
            used_cur = set()
            for x in candidates:
                if x in used_cur:
                    continue
                if _can_add(cur, x):
                    cur.append(x)
                    used_cur.add(x)
                if len(cur) >= len(best):
                    # Early check: if we already have as many, continue to try for more
                    pass
            if len(cur) > len(best):
                best = list(cur)
    
    # Final: try to add more elements to best by trying all unused
    if time.time() < deadline - 2:
        used_best = set(best)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, 257):
                if x in used_best:
                    continue
                if _can_add(best, x):
                    best.append(x)
                    used_best.add(x)
                    improved = True
                    break
    
    # Ensure we return Python ints in a list
    result = [int(x) for x in best]
    return result

# EVOLVE-BLOCK-END
