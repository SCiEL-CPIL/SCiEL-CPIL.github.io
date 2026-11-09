"""Rewritten program: randomized greedy with backtracking for divisibility-incomparable subset sums."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _subset_sums_valid(elements):
    """Check that all nonempty subset sums are unique and no two divide each other."""
    n = len(elements)
    sums = set()
    # Generate all nonempty subset sums
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return False  # duplicate sum -> invalid
        # Check divisibility against all existing sums
        for t in sums:
            if s % t == 0 or t % s == 0:
                return False
        sums.add(s)
    return True

def _is_valid_set(elements):
    """Full validity check: distinct ints in 1..256, subset sums unique and divisibility-incomparable."""
    if len(elements) != len(set(elements)):
        return False
    if any(e < 1 or e > 256 for e in elements):
        return False
    return _subset_sums_valid(elements)

def _try_extend(current, candidates, target_size):
    """Try to extend current set to target_size using backtracking."""
    if len(current) == target_size:
        return current[:]
    for i, cand in enumerate(candidates):
        new_set = current + [cand]
        # Quick check: subset sums of new_set
        if _subset_sums_valid(new_set):
            result = _try_extend(new_set, candidates[i+1:], target_size)
            if result is not None:
                return result
    return None

def _greedy_build(max_elements, rng, deadline):
    """Greedy: try to build a set by adding elements one at a time."""
    # Start with larger numbers to reduce divisibility conflicts
    # Strategy: pick numbers in a range that makes subset sums fall in a "safe" band
    best = []
    
    # Try multiple random starts
    for attempt in range(200):
        if time.time() > deadline:
            break
        rng2 = random.Random(rng.random())
        
        # Pick a starting "scale" - larger numbers tend to work better
        # Try to build from a base set
        current = []
        used = set()
        
        # Generate candidate pool: focus on numbers in a high range
        # e.g., 100-256, where subset sums are large and less likely to divide
        pool = list(range(90, 257))
        rng2.shuffle(pool)
        
        for cand in pool:
            if len(current) >= max_elements:
                break
            if time.time() > deadline:
                break
            if cand in used:
                continue
            new_set = current + [cand]
            if _subset_sums_valid(new_set):
                current.append(cand)
                used.add(cand)
        
        if len(current) > len(best):
            best = current[:]
    
    return best

def _local_search(initial, rng, deadline):
    """Try to improve via swaps and additions."""
    current = initial[:]
    used = set(current)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Try adding a random element
        if len(current) < 30:
            candidates = [x for x in range(1, 257) if x not in used]
            if candidates:
                cand = rng.choice(candidates)
                new_set = current + [cand]
                if _subset_sums_valid(new_set):
                    current.append(cand)
                    used.add(cand)
                    continue
        
        # Try swapping: remove one, add another
        if current:
            rem_idx = rng.randrange(len(current))
            rem = current[rem_idx]
            candidates = [x for x in range(1, 257) if x not in used]
            if candidates:
                add = rng.choice(candidates)
                new_set = [e for i, e in enumerate(current) if i != rem_idx] + [add]
                if _subset_sums_valid(new_set):
                    current = new_set
                    used.discard(rem)
                    used.add(add)
                    continue
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (limit is 90s)
    
    rng = random.Random(42)
    
    best_set = []
    
    # Phase 1: Greedy construction with multiple randomizations
    for max_size in range(10, 4, -1):  # Try to build up to 10 elements
        if time.time() > deadline:
            break
        result = _greedy_build(max_size, rng, deadline)
        if len(result) > len(best_set):
            best_set = result
        if len(best_set) >= 10:
            break
    
    # Phase 2: Local search to improve
    if best_set:
        improved = _local_search(best_set, rng, deadline)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Phase 3: Try different base ranges
    for base_range in [(50, 200), (100, 256), (75, 220), (120, 256)]:
        if time.time() > deadline:
            break
        lo, hi = base_range
        for attempt in range(50):
            if time.time() > deadline:
                break
            rng3 = random.Random(rng.random())
            current = []
            used = set()
            pool = list(range(lo, hi + 1))
            rng3.shuffle(pool)
            
            for cand in pool:
                if len(current) >= 12:
                    break
                if time.time() > deadline:
                    break
                if cand in used:
                    continue
                new_set = current + [cand]
                if _subset_sums_valid(new_set):
                    current.append(cand)
                    used.add(cand)
            
            if len(current) > len(best_set):
                best_set = current
    
    # Ensure return type is list of Python ints
    result = [int(x) for x in best_set]
    
    # Final validation
    if not _is_valid_set(result):
        # Fallback to known valid small set
        result = [31, 32, 34]
    
    return result

# EVOLVE-BLOCK-END
