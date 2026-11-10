import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill
    
    def is_valid(s):
        """Check if subset sums with different cardinalities are all distinct."""
        n = len(s)
        # For each cardinality k, compute all subset sums
        # Use bitset or set approach
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        for x in s:
            # Update in reverse to avoid using x multiple times
            for k in range(n, 0, -1):
                if k > 0:
                    for prev_sum in list(sums_by_card[k - 1]):
                        sums_by_card[k].add(prev_sum + x)
        # Check: for any k1 != k2, sums_by_card[k1] and sums_by_card[k2] must be disjoint
        for k1 in range(n + 1):
            for k2 in range(k1 + 1, n + 1):
                if sums_by_card[k1] & sums_by_card[k2]:
                    return False
        return True
    
    # Try arithmetic progressions from largest n downward
    # {a, a+1, ..., a+n-1} where a+n-1 <= 256
    best = list(range(226, 257))  # fallback: n=31, known valid
    
    for n in range(35, 20, -1):
        if time.time() > deadline:
            break
        # Try various starting values
        for a in range(256 - n + 1, 0, -1):
            if time.time() > deadline:
                break
            s = list(range(a, a + n))
            if is_valid(s):
                best = s
                break
        if len(best) >= n:
            break
    
    # Try to extend best by adding elements
    best_set = set(best)
    for candidate in range(1, 257):
        if time.time() > deadline:
            break
        if candidate in best_set:
            continue
        trial = best + [candidate]
        if is_valid(trial):
            best = trial
            best_set.add(candidate)
    
    # Ensure Python ints and list
    return [int(x) for x in best]
