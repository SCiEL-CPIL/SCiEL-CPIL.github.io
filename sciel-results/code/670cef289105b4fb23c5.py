import time
import random

START_TIME = time.time()
DEADLINE = 60.0  # seconds

def elapsed():
    return time.time() - START_TIME

def is_valid_set(s):
    """Check if a set satisfies the cardinality-separated subset sum condition."""
    n = len(s)
    if n == 0:
        return True
    max_sum = sum(s)
    # For each cardinality k, store the set of all subset sums of size k
    # Use bitsets for efficiency
    # sums_by_card[k] = set of sums achievable with exactly k elements
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for i, val in enumerate(s):
        # Update in reverse to avoid using the same element twice
        for k in range(i + 1, 0, -1):
            for old_sum in sums_by_card[k - 1]:
                sums_by_card[k].add(old_sum + val)
    
    # Check that sums of different cardinalities are disjoint
    all_sums = []
    for k in range(n + 1):
        all_sums.append(sums_by_card[k])
    
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if all_sums[i] & all_sums[j]:
                return False
    return True

def greedy_construct(order, deadline_margin=5.0):
    """Greedily build a valid set by adding elements in the given order."""
    s = []
    for val in order:
        if elapsed() > DEADLINE - deadline_margin:
            break
        candidate = s + [val]
        if is_valid_set(candidate):
            s = candidate
    return s

def construct():
    if elapsed() > DEADLINE - 2.0:
        # Fallback to known valid set
        return list(range(226, 257))
    
    best = list(range(226, 257))  # Known valid size-31 set
    best_size = 31
    
    # Try different orderings to find larger sets
    # Strategy: try descending, ascending, and random permutations
    candidates = list(range(1, 257))
    
    # Try descending order
    if elapsed() < DEADLINE - 10.0:
        s = greedy_construct(sorted(candidates, reverse=True), deadline_margin=5.0)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Try ascending order
    if elapsed() < DEADLINE - 10.0:
        s = greedy_construct(sorted(candidates), deadline_margin=5.0)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Try several random orderings
    random.seed(42)
    for trial in range(20):
        if elapsed() > DEADLINE - 10.0:
            break
        random_order = candidates[:]
        random.shuffle(random_order)
        s = greedy_construct(random_order, deadline_margin=5.0)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Final check: ensure best is valid
    if not is_valid_set(best):
        return list(range(226, 257))
    
    # Ensure all elements are Python ints in a list
    return [int(x) for x in best]
