"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _all_subset_sums(elements):
    """Return set of all nonempty subset sums."""
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s > 0:
            sums.add(s)
    return sums

def _is_valid(elements):
    """Check if the set of elements satisfies the divisibility-incomparability constraint."""
    if len(elements) == 0:
        return True
    sums = _all_subset_sums(elements)
    if len(sums) != (1 << len(elements)) - 1:
        return False  # duplicate subset sums
    # Check no two distinct sums divide one another
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % sorted_sums[i] == 0:
                return False
    return True

def _is_valid_fast(elements):
    """Faster validity check using incremental subset sums."""
    n = len(elements)
    if n == 0:
        return True
    # Generate all subset sums
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return False
        sums.add(s)
    # Check divisibility
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        si = sorted_sums[i]
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % si == 0:
                return False
    return True

def _greedy_build(start_elements, candidates, target_size, deadline):
    """Greedy: add candidates that maintain validity."""
    current = list(start_elements)
    if len(current) >= target_size:
        return current
    
    remaining = [c for c in candidates if c not in current]
    for c in remaining:
        if time.time() > deadline:
            break
        trial = current + [c]
        if _is_valid_fast(trial):
            current = trial
            if len(current) >= target_size:
                break
    return current

def _local_search(elements, candidates, target_size, deadline):
    """Local search: try swap-then-add moves."""
    current = list(elements)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding a new element
        for c in candidates:
            if c in current:
                continue
            trial = current + [c]
            if _is_valid_fast(trial):
                current = trial
                improved = True
                if len(current) >= target_size:
                    return current
                break
        if len(current) >= target_size:
            return current
        # Try swap: remove one, add one or two
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for c in candidates:
                if c in current:
                    continue
                # Swap: remove current[i], add c
                trial = [x for j, x in enumerate(current) if j != i] + [c]
                if _is_valid_fast(trial):
                    current = trial
                    improved = True
                    break
                # Swap+add: remove current[i], add c and another
                for c2 in candidates:
                    if c2 == c or c2 in current:
                        continue
                    trial2 = [x for j, x in enumerate(current) if j != i] + [c, c2]
                    if _is_valid_fast(trial2):
                        current = trial2
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return current

def construct():
    deadline = time.time() + 80  # 80s to leave margin
    
    # Candidate pool: numbers in 1..256
    # Start with small numbers that are likely to work
    # Superincreasing-like starts help with distinct sums
    base_sets = [
        [2, 3, 5, 9, 17, 33, 65, 129],
        [3, 5, 9, 17, 33, 65, 129],
        [2, 3, 5, 9, 17, 33, 65],
        [3, 5, 11, 23, 47, 95, 191],
        [2, 3, 7, 15, 31, 63, 127],
        [4, 5, 7, 11, 19, 37, 73],
        [3, 4, 7, 13, 25, 49, 97],
        [5, 7, 11, 19, 37, 73, 145],
    ]
    
    candidates = list(range(1, 257))
    
    best = []
    
    # Try greedy from each base
    for base in base_sets:
        if time.time() > deadline:
            break
        # Filter base to valid
        if not _is_valid_fast(base):
            # Trim to valid prefix
            valid_base = []
            for x in base:
                trial = valid_base + [x]
                if _is_valid_fast(trial):
                    valid_base = trial
                else:
                    break
            base = valid_base
        
        result = _greedy_build(base, candidates, 30, deadline)
        if len(result) > len(best):
            best = result
        
        # Local search from greedy result
        result_ls = _local_search(result, candidates, 30, deadline)
        if len(result_ls) > len(best):
            best = result_ls
    
    # Try random restarts
    import random
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        # Random starting set of size 3-5
        start_size = rng.randint(3, 5)
        start = rng.sample(candidates, start_size)
        start.sort()
        
        if not _is_valid_fast(start):
            continue
        
        result = _greedy_build(start, candidates, 30, deadline)
        if len(result) > len(best):
            best = result
        
        result_ls = _local_search(result, candidates, 30, deadline)
        if len(result_ls) > len(best):
            best = result_ls
    
    # Final validation
    if best and _is_valid_fast(best):
        return list(best)
    
    # Fallback: return a known valid small set
    fallback = [2, 3, 5]
    if _is_valid_fast(fallback):
        return list(fallback)
    
    return [2, 3, 5]

# EVOLVE-BLOCK-END
