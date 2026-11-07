"""Rewritten program: search for large valid sets with deadline guard."""
import time
import itertools
import random

DEADLINE_SECONDS = 85.0  # safety margin before 90s kill

def _is_valid_set(nums):
    """Check if set of distinct ints in 1..256 has no two distinct subset sums dividing each other."""
    n = len(nums)
    if n == 0:
        return True
    # Generate all nonempty subset sums
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    # Check no two distinct sums divide each other
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % sorted_sums[i] == 0:
                return False
    return True

def _greedy_build(start_set, max_size=30, deadline=None):
    """Try to extend start_set greedily."""
    current = list(start_set)
    used = set(current)
    # Precompute subset sums of current set
    def get_sums(lst):
        n = len(lst)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += lst[i]
            sums.add(s)
        return sums
    
    # Try adding elements from 1..256 not in used
    candidates = [x for x in range(1, 257) if x not in used]
    for x in candidates:
        if len(current) >= max_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        test_set = current + [x]
        if _is_valid_set(test_set):
            current = test_set
            used.add(x)
    return current

def _random_search(deadline, best_size=3):
    """Randomized search to find larger valid sets."""
    best = [31, 32, 34]  # known valid
    best_size = len(best)
    
    # Try random subsets of increasing size
    random.seed(42)
    
    # First, try to extend the known set
    extended = _greedy_build(best, max_size=30, deadline=deadline)
    if len(extended) > best_size:
        best = extended
        best_size = len(extended)
    
    # Try random initial sets and extend
    attempts = 0
    while time.time() < deadline and best_size < 30:
        attempts += 1
        # Try random sets of size 4-8
        size = random.randint(4, min(10, 256))
        rand_set = random.sample(range(1, 257), size)
        if _is_valid_set(rand_set):
            extended = _greedy_build(rand_set, max_size=30, deadline=deadline)
            if len(extended) > best_size:
                best = extended
                best_size = len(extended)
        
        # Also try structured sets: multiples of a base
        if attempts % 5 == 0:
            base = random.randint(5, 20)
            struct_set = [base * i for i in range(1, 20) if base * i <= 256]
            if len(struct_set) >= 4:
                if _is_valid_set(struct_set[:4]):
                    extended = _greedy_build(struct_set[:4], max_size=30, deadline=deadline)
                    if len(extended) > best_size:
                        best = extended
                        best_size = len(extended)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with known valid set
    best = [31, 32, 34]
    
    # Try to improve
    if time.time() < deadline:
        improved = _random_search(deadline, best_size=3)
        if len(improved) > len(best):
            best = improved
    
    # Ensure return type is list of Python ints
    result = [int(x) for x in best]
    return result
