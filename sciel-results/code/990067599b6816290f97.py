"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _subset_sums_conflict(sums_set):
    """Check if any two distinct values in sums_set divide one another.
    Also check if there are duplicate sums (handled by caller)."""
    s = sorted(sums_set)
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            if s[j] % s[i] == 0:
                return True
    return False

def _validate_set(arr):
    """Check if arr is valid: distinct integers in 1..256, all nonempty subset sums are unique and divisibility-incomparable."""
    if len(set(arr)) != len(arr):
        return False
    for x in arr:
        if not (isinstance(x, int) and 1 <= x <= 256):
            return False
    n = len(arr)
    # Compute all subset sums
    from collections import defaultdict
    sum_count = defaultdict(int)
    # Use bitmask enumeration
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        sum_count[s] += 1
    # Check for duplicate sums
    for s, cnt in sum_count.items():
        if cnt > 1:
            return False
    # Check divisibility among distinct sums
    sums_list = sorted(sum_count.keys())
    n_sums = len(sums_list)
    for i in range(n_sums):
        for j in range(i + 1, n_sums):
            if sums_list[j] % sums_list[i] == 0:
                return False
    return True

def _try_add(arr, candidate, deadline):
    """Try to add candidate to arr. Return True if valid."""
    if time.time() > deadline:
        return False
    new_arr = arr + [candidate]
    n = len(new_arr)
    # Quick check: new subset sums
    # All subset sums of new_arr = subset sums of arr (without candidate) + candidate + subset sums of arr
    # But we need to check divisibility and uniqueness
    
    # For efficiency, compute all subset sums of arr first
    sum_count = {}
    for mask in range(1, 1 << (n - 1)):
        s = 0
        for i in range(n - 1):
            if mask & (1 << i):
                s += new_arr[i]
        if s in sum_count:
            return False
        sum_count[s] = 1
    
    # Now add candidate: new sums are old sums + candidate
    # And the candidate itself
    old_sums = list(sum_count.keys())
    
    # Check candidate itself
    if candidate in sum_count:
        return False
    # Check candidate against old sums
    for s in old_sums:
        if candidate % s == 0 or s % candidate == 0:
            return False
    
    # Check new sums (old + candidate) against all existing sums (old + candidate + candidate itself)
    new_sums = {}
    for s in old_sums:
        ns = s + candidate
        if ns in new_sums or ns in sum_count or ns == candidate:
            return False
        new_sums[ns] = 1
    
    # Check new sums against each other and against old sums and candidate
    all_sums = set(sum_count.keys()) | set(new_sums.keys()) | {candidate}
    sums_list = sorted(all_sums)
    n_sums = len(sums_list)
    for i in range(n_sums):
        for j in range(i + 1, n_sums):
            if sums_list[j] % sums_list[i] == 0:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10 second safety margin
    
    best = [31, 32, 34]  # seed
    
    # Heuristic construction: build set by adding elements one at a time
    # Strategy: try to find a large set using randomized greedy with backtracking
    
    # First, let's try a systematic search with increasing size
    # Use random restarts
    
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Randomly pick a starting set and try to grow it
        # Start with a small random valid set
        current = []
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Try to build up greedily
        for cand in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 30:
                break
            if _try_add(current, cand, deadline):
                current.append(cand)
        
        if len(current) > len(best):
            best = current[:]
            if len(best) >= 30:
                break
        
        # Also try to improve current best by removing one and adding another
        if len(best) > 0 and time.time() < deadline:
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                # Try replacing best[i] with a random candidate
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    new_cand = random.randint(1, 256)
                    if new_cand in best:
                        continue
                    trial = best[:i] + best[i+1:] + [new_cand]
                    if _validate_set(trial) and len(trial) == len(best):
                        # Same size, but might help future growth
                        # Actually we want to grow, so let's try growing from here
                        # This is getting complex, let's just keep the best found
                        pass
    
    # Final validation
    if not _validate_set(best):
        return [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
