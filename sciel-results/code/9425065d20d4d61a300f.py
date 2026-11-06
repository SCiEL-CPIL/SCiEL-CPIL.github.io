import time
import random
import numpy as np

DEADLINE = 145.0  # safety margin before 150s kill

def _counts_from_set(s):
    """Compute sum counts for a set s (list of ints). Returns dict sum->count."""
    s = list(s)
    n = len(s)
    counts = {}
    # Use numpy for speed: compute all pairwise sums
    if n == 0:
        return counts
    arr = np.array(s, dtype=np.int64)
    # All pairwise sums a+b where a<=b
    # We can do this efficiently
    sums = np.zeros(n*n, dtype=np.int64)
    for i in range(n):
        sums[i*n:(i+1)*n] = arr[i] + arr
    # Only count pairs where i <= j (a <= b)
    for i in range(n):
        for j in range(i, n):
            s_val = int(arr[i] + arr[j])
            counts[s_val] = counts.get(s_val, 0) + 1
    return counts

def _counts_from_set_fast(s):
    """Faster count computation using numpy vectorization."""
    s = list(s)
    n = len(s)
    if n == 0:
        return {}
    arr = np.array(s, dtype=np.int64)
    # Compute all sums
    all_sums = arr[:, None] + arr[None, :]
    # Mask for i <= j
    mask = np.triu(np.ones((n, n), dtype=bool))
    valid_sums = all_sums[mask]
    # Count using np.bincount
    min_val = 2  # minimum possible sum (1+1)
    max_val = 2000  # maximum possible sum (1000+1000)
    counts_arr = np.bincount(valid_sums - min_val, minlength=max_val - min_val + 1)
    counts = {}
    for i, c in enumerate(counts_arr):
        if c > 0:
            counts[min_val + i] = int(c)
    return counts

def _is_valid(s):
    """Check if set s satisfies B2[2] property: every sum has at most 2 representations."""
    counts = _counts_from_set_fast(s)
    for c in counts.values():
        if c > 2:
            return False
    return True

def _max_count(s):
    """Return the maximum number of representations of any sum."""
    counts = _counts_from_set_fast(s)
    if not counts:
        return 0
    return max(counts.values())

def _greedy_construct():
    """Original greedy algorithm."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        valid = True
        for s_val in new_sums:
            trial[s_val] = trial.get(s_val, 0) + 1
            if trial[s_val] > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            counts = trial
    return chosen

def _local_search(s, time_limit=10.0):
    """Local search with swap moves to improve the set."""
    s = list(s)
    best_s = s[:]
    best_size = len(s)
    start = time.time()
    
    # Precompute counts
    counts = _counts_from_set_fast(s)
    
    # Try to add elements
    improvement = True
    while improvement and time.time() - start < time_limit:
        improvement = False
        # Try adding each candidate
        for x in range(1, 1001):
            if x in s:
                continue
            new_sums = [x + y for y in s] + [2 * x]
            trial = dict(counts)
            valid = True
            for s_val in new_sums:
                trial[s_val] = trial.get(s_val, 0) + 1
                if trial[s_val] > 2:
                    valid = False
                    break
            if valid:
                s.append(x)
                counts = trial
                best_s = s[:]
                best_size = len(s)
                improvement = True
                if time.time() - start > time_limit:
                    return best_s
        # If no additions, try swaps
        if not improvement:
            swapped = False
            for i in range(len(s)):
                for x in range(1, 1001):
                    if x in s:
                        continue
                    # Try swapping s[i] with x
                    new_s = s[:]
                    new_s[i] = x
                    new_s.sort()
                    if _is_valid(new_s):
                        s = new_s
                        counts = _counts_from_set_fast(s)
                        best_s = s[:]
                        best_size = len(s)
                        swapped = True
                        improvement = True
                        if time.time() - start > time_limit:
                            return best_s
                        break
                if swapped and time.time() - start > time_limit:
                    return best_s
    return best_s

def _random_greedy(seed):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    order = list(range(1, 1001))
    rng.shuffle(order)
    chosen = []
    counts = {}
    for x in order:
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        valid = True
        for s_val in new_sums:
            trial[s_val] = trial.get(s_val, 0) + 1
            if trial[s_val] > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            counts = trial
    return chosen

def construct():
    start_time = time.time()
    
    # Start with greedy
    best_s = _greedy_construct()
    best_size = len(best_s)
    
    # Time remaining for local search
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        return best_s
    
    # Local search on the greedy result
    best_s = _local_search(best_s, time_limit=min(20, DEADLINE - (time.time() - start_time) - 5))
    best_size = len(best_s)
    
    # Try random greedy + local search if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        for seed in range(10):
            if time.time() - start_time > DEADLINE - 25:
                break
            s = _random_greedy(seed)
            if len(s) > best_size:
                best_s = s
                best_size = len(s)
            # Quick local search
            remaining = DEADLINE - (time.time() - start_time) - 5
            if remaining > 3:
                s2 = _local_search(s, time_limit=min(3, remaining))
                if len(s2) > best_size:
                    best_s = s2
                    best_size = len(s2)
    
    return best_s
