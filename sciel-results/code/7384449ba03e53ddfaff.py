"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard
_DEADLINE_SECONDS = 80.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _check_deadline():
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _all_subset_sums_distinct(reciprocals):
    """Check if all 2^n subset sums of reciprocals are distinct."""
    n = len(reciprocals)
    if n > 25:
        return False
    sums = set()
    # Use bit-based enumeration
    for mask in range(1 << n):
        s = 0.0
        for i in range(n):
            if mask & (1 << i):
                s += reciprocals[i]
        # Use a tolerance-based approach: store as rounded integer
        key = round(s * 10**12)
        if key in sums:
            return False
        sums.add(key)
    return True

def _verify_with_fractions(subset):
    """Verify using exact fractions for small subsets."""
    from fractions import Fraction
    n = len(subset)
    if n > 25:
        return False
    reciprocals = [Fraction(1, x) for x in subset]
    sums = set()
    for mask in range(1 << n):
        s = Fraction(0)
        for i in range(n):
            if mask & (1 << i):
                s += reciprocals[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _greedy_search(max_size, trial_seed=0):
    """Greedy search: try to build a large dissociated reciprocal set."""
    rng = random.Random(trial_seed)
    candidates = list(range(1, 23))
    
    # Try to find a set of size max_size
    # Use a heuristic: prefer larger numbers (smaller reciprocals) to reduce collision risk
    # but also try various orderings
    
    for order_seed in range(10):
        if _check_deadline():
            break
        rng2 = random.Random(trial_seed * 100 + order_seed)
        ordered = candidates[:]
        # Mix in some randomization
        for i in range(len(ordered) - 1, 0, -1):
            j = rng2.randint(0, i)
            ordered[i], ordered[j] = ordered[j], ordered[i]
        
        # Greedy: start empty, try to add each element
        current = []
        current_sums = {0.0}
        
        for x in ordered:
            if len(current) >= max_size:
                break
            if _check_deadline():
                break
            recip = 1.0 / x
            # Check if adding recip creates a duplicate sum
            new_sums = set()
            valid = True
            for s in list(current_sums):
                ns = s + recip
                # Check against existing sums
                if ns in current_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                current.append(x)
                current_sums.update(new_sums)
        
        if len(current) >= max_size:
            # Verify with exact fractions
            if _verify_with_fractions(sorted(current)):
                return sorted(current)
    
    return None

def _random_subset_search(target_size, num_trials, seed_offset=0):
    """Randomly sample subsets of target_size and check."""
    for trial in range(num_trials):
        if _check_deadline():
            break
        seed = seed_offset + trial
        rng = random.Random(seed)
        subset = rng.sample(range(1, 23), target_size)
        subset = sorted(subset)
        if _verify_with_fractions(subset):
            return subset
    return None

def _backtrack_search(target_size, seed_offset=0):
    """Backtracking search with pruning for target_size."""
    from fractions import Fraction
    
    def backtrack(start_idx, current, current_sums):
        if len(current) == target_size:
            return current[:]
        if _check_deadline():
            return None
        
        remaining = 23 - 1 - start_idx + 1  # numbers from start_idx to 22
        need = target_size - len(current)
        if remaining < need:
            return None
        
        for i in range(start_idx, 23):
            if len(current) == target_size:
                break
            if _check_deadline():
                return None
            x = i
            recip = Fraction(1, x)
            # Check if adding recip creates a duplicate sum
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + recip
                if ns in current_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                current.append(x)
                current_sums.update(new_sums)
                result = backtrack(i + 1, current, current_sums)
                if result is not None:
                    return result
                current.pop()
                for ns in new_sums:
                    current_sums.discard(ns)
        return None
    
    result = backtrack(1, [], {Fraction(0)})
    if result is not None:
        return sorted(result)
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best = None
    
    # First, try to find a large set using backtracking
    # Start with size 12 and work down if needed
    for target_size in [12, 11, 10, 9, 8, 7, 6, 5]:
        if _check_deadline():
            break
        
        # Try backtracking with a few different starting strategies
        result = _backtrack_search(target_size, seed_offset=0)
        if result is not None:
            best = result
            break
    
    if best is None:
        # Fallback: try greedy
        for target_size in [10, 9, 8, 7, 6, 5]:
            if _check_deadline():
                break
            result = _greedy_search(target_size, trial_seed=42)
            if result is not None:
                best = result
                break
    
    if best is None:
        # Last resort: random sampling
        for target_size in [8, 7, 6, 5, 4]:
            if _check_deadline():
                break
            result = _random_subset_search(target_size, num_trials=1000, seed_offset=12345)
            if result is not None:
                best = result
                break
    
    if best is None:
        # Absolute fallback: known valid set
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Final verification
    if not _verify_with_fractions(best):
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best
# EVOLVE-BLOCK-END
