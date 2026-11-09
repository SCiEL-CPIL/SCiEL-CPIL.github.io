"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _subset_sums_with_counts(elements):
    """Return dict: sum -> count of subsets producing that sum."""
    n = len(elements)
    counts = {}
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        counts[s] = counts.get(s, 0) + 1
    return counts

def _check_divisibility(sums_list):
    """Check no two distinct sums divide each other."""
    s = sorted(sums_list)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[j] % s[i] == 0:
                return False
    return True

def _is_valid(elements):
    """Full validity check for a set of elements."""
    if len(elements) != len(set(elements)):
        return False
    if any(x < 1 or x > 256 for x in elements):
        return False
    counts = _subset_sums_with_counts(elements)
    if any(v > 1 for v in counts.values()):
        return False
    return _check_divisibility(list(counts.keys()))

def _try_add_incremental(current_sums, current_counts, candidate, deadline):
    """
    Try adding candidate to a set. 
    current_sums: set of all valid subset sums so far
    current_counts: dict sum -> count
    Returns (new_sums, new_counts, True) if valid, else None.
    """
    if time.time() > deadline:
        return None
    
    new_sums = set(current_sums)
    new_counts = dict(current_counts)
    
    # For each existing sum, add candidate
    # Also the candidate itself
    existing_sums_list = list(current_sums)
    
    # Check if candidate alone is valid (it's new, so count 1)
    # But we need to check if candidate divides any existing sum or is divided by any
    # And if (candidate + s) divides any existing sum or is divided by any, etc.
    
    # Efficient approach: generate all new sums, check for collisions, then check divisibility
    
    candidate_sum = candidate
    new_sum_values = [candidate_sum]
    
    for s in existing_sums_list:
        new_sum_values.append(s + candidate)
    
    # Check for duplicates within new sums
    for ns in new_sum_values:
        if ns in new_sums:
            # Collision: two different subsets produce same sum
            return None
        # Check divisibility: ns vs all existing sums
        for es in new_sums:
            if ns != es:
                if ns % es == 0 or es % ns == 0:
                    return None
    
    # Check divisibility among new sums themselves
    for i in range(len(new_sum_values)):
        for j in range(i + 1, len(new_sum_values)):
            a = new_sum_values[i]
            b = new_sum_values[j]
            if a != b and (a % b == 0 or b % a == 0):
                return None
    
    # All checks passed, update
    for ns in new_sum_values:
        new_sums.add(ns)
        new_counts[ns] = new_counts.get(ns, 0) + 1
    
    return (new_sums, new_counts, True)

def _greedy_build(candidates, deadline, max_size=30):
    """Greedy: add candidates in given order if valid."""
    current_elements = []
    current_sums = set()
    current_counts = {}
    
    for c in candidates:
        if time.time() > deadline:
            break
        if len(current_elements) >= max_size:
            break
        
        result = _try_add_incremental(current_sums, current_counts, c, deadline)
        if result is not None:
            current_elements.append(c)
            current_sums, current_counts, _ = result
    
    return current_elements

def _local_search_improve(best_set, deadline):
    """Try to improve by swapping or adding elements."""
    current_elements = list(best_set)
    
    # Rebuild sums and counts
    current_sums = set()
    current_counts = {}
    n = len(current_elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += current_elements[i]
        current_sums.add(s)
        current_counts[s] = current_counts.get(s, 0) + 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a new element
        for c in range(1, 257):
            if time.time() > deadline:
                break
            if c in current_elements:
                continue
            result = _try_add_incremental(current_sums, current_counts, c, deadline)
            if result is not None:
                current_elements.append(c)
                current_sums, current_counts, _ = result
                improved = True
                break
        
        # Try swapping: remove one, add another
        if not improved:
            for i in range(len(current_elements)):
                if time.time() > deadline:
                    break
                for c in range(1, 257):
                    if time.time() > deadline:
                        break
                    if c in current_elements:
                        continue
                    
                    # Try swap: remove current_elements[i], add c
                    # This requires rebuilding sums, which is expensive
                    # Skip for now, focus on additions
                    pass
    
    return current_elements

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_set = [31, 32, 34]
    
    # Strategy 1: Greedy with large numbers first (descending)
    cands_desc = list(range(256, 0, -1))
    result = _greedy_build(cands_desc, deadline)
    if len(result) > len(best_set):
        best_set = result
    
    # Strategy 2: Multiple randomized greedy attempts
    for attempt in range(2000):
        if time.time() > deadline:
            break
        cands = list(range(1, 257))
        random.shuffle(cands)
        result = _greedy_build(cands, deadline)
        if len(result) > len(best_set):
            best_set = result
            if len(best_set) >= 30:
                break
    
    # Strategy 3: Local search improvement
    if time.time() < deadline:
        improved_set = _local_search_improve(best_set, deadline)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Strategy 4: Try specific known-good starting points and extend
    if time.time() < deadline:
        starting_sets = [
            [128, 192, 224],
            [100, 150, 200],
            [64, 96, 128, 160],
            [80, 120, 160, 200, 240],
            [50, 80, 110, 140, 170, 200],
            [16, 32, 64, 128],
            [24, 48, 96, 192],
            [36, 72, 144],
            [48, 96, 192],
            [64, 128, 192],
        ]
        for ss in starting_sets:
            if time.time() > deadline:
                break
            if not _is_valid(ss):
                continue
            # Try to extend from this starting point
            cands = [c for c in range(1, 257) if c not in ss]
            random.shuffle(cands)
            result = _greedy_build(ss + cands, deadline)
            if len(result) > len(best_set):
                best_set = result
                if len(best_set) >= 30:
                    break
    
    # Final local search
    if time.time() < deadline - 5:
        improved_set = _local_search_improve(best_set, deadline)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
