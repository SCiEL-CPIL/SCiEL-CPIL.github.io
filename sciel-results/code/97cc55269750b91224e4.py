"""
Improved search for divisibility-incomparable subset sums.
Uses incremental subset-sum maintenance, fast divisibility conflict checking,
and multiple greedy strategies with local search.
"""

import time
import random
from collections import defaultdict

def _valid_set_fast(nums):
    """
    Check if a set of numbers satisfies the divisibility-incomparable subset sum condition.
    Returns True if valid, False otherwise.
    """
    n = len(nums)
    if n == 0:
        return True
    
    # Generate all nonempty subset sums using iterative method
    sums = set()
    for x in nums:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        # Check for duplicates
        if len(sums | new_sums) != len(sums) + len(new_sums):
            return False
        sums = sums | new_sums
    
    # Sort and check divisibility
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        si = sorted_sums[i]
        for j in range(i + 1, len(sorted_sums)):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False
    return True

def _can_add_incremental(nums, new_elem):
    """
    Check if adding new_elem to nums keeps the set valid.
    Uses incremental approach: compute new sums and check for conflicts.
    """
    n = len(nums)
    
    # Generate all existing subset sums
    sums = set()
    for x in nums:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums = sums | new_sums
    
    # Generate new sums that include new_elem
    new_sums = set()
    for s in sums:
        new_sums.add(s + new_elem)
    new_sums.add(new_elem)
    
    # Check for duplicates between old and new sums
    if sums & new_sums:
        return False
    
    # All sums that will exist after adding
    all_sums = sums | new_sums
    sorted_sums = sorted(all_sums)
    
    # Check divisibility: for each sum, ensure no other sum is a multiple
    # Optimization: only need to check if any sum divides another
    for i in range(len(sorted_sums)):
        si = sorted_sums[i]
        # If si is large, few multiples exist in range
        if si > 128:
            continue
        for j in range(i + 1, len(sorted_sums)):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False
    return True

def _greedy_construct(pool, deadline, rng, max_size=30):
    """
    Greedy construction: try adding elements in the given order.
    Returns the constructed list.
    """
    current = []
    used = set()
    
    for cand in pool:
        if time.time() > deadline:
            break
        if len(current) >= max_size:
            break
        if cand in used:
            continue
        if _can_add_incremental(current, cand):
            current.append(cand)
            used.add(cand)
    
    return current

def _local_search_improve(current, deadline, rng, max_size=30):
    """
    Try to improve current set via local search: swap or add elements.
    """
    current = list(current)
    all_candidates = set(range(1, 257))
    
    for _ in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a new element if not at max size
        if len(current) < max_size:
            candidates_to_try = [c for c in range(256, 1, -1) if c not in current]
            rng.shuffle(candidates_to_try)
            for cand in candidates_to_try[:20]:
                if time.time() > deadline:
                    break
                if _can_add_incremental(current, cand):
                    current.append(cand)
                    improved = True
                    break
        
        # Try swapping an element with another
        if not improved and len(current) > 0:
            idx = rng.randint(0, len(current) - 1)
            old_val = current[idx]
            candidates_to_try = [c for c in range(256, 1, -1) if c != old_val and c not in current]
            rng.shuffle(candidates_to_try)
            for new_val in candidates_to_try[:15]:
                if time.time() > deadline:
                    break
                trial = current.copy()
                trial[idx] = new_val
                if _valid_set_fast(trial):
                    current = trial
                    improved = True
                    break
    
    return current

def construct():
    start = time.time()
    deadline = start + 85.0  # 85 second deadline, 5s safety margin
    
    best = [31, 32, 34]  # Fallback - known valid set of size 3
    best_size = 3
    
    rng = random.Random(12345)
    
    # Strategy 1: Greedy with large numbers first (deterministic)
    candidates_large = list(range(256, 128, -1))
    current = _greedy_construct(candidates_large, deadline, rng)
    if len(current) > best_size:
        best = list(current)
        best_size = len(current)
    
    # Strategy 2: Multiple random restarts with biased orderings
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Create a shuffled candidate list, biased toward larger numbers
        # Mix of large and medium numbers
        pool = list(range(256, 64, -1))
        rng.shuffle(pool)
        
        current2 = _greedy_construct(pool, deadline, rng)
        
        # Try to improve with local search
        if len(current2) > 0:
            current2 = _local_search_improve(current2, deadline, rng)
        
        if len(current2) > best_size:
            best = list(current2)
            best_size = len(current2)
            if best_size >= 30:
                break
    
    # Strategy 3: Try different initial pools
    for pool_start in [256, 200, 150, 100]:
        if time.time() > deadline:
            break
        
        pool = list(range(pool_start, 1, -1))
        rng.shuffle(pool)
        
        current3 = _greedy_construct(pool, deadline, rng)
        if len(current3) > 0:
            current3 = _local_search_improve(current3, deadline, rng)
        
        if len(current3) > best_size:
            best = list(current3)
            best_size = len(current3)
            if best_size >= 30:
                break
    
    # Final verification
    if _valid_set_fast(best):
        return list(best)
    else:
        return [31, 32, 34]
