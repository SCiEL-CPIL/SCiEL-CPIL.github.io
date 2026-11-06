import time
import random
from itertools import combinations
from bisect import bisect_left

# LCM of 1..22
_LCM = 232792560
_WEIGHTS = {x: _LCM // x for x in range(1, 23)}

def _can_add(sorted_sums, w):
    """Check if adding weight w to the set of subset sums causes no collision."""
    for i, s in enumerate(sorted_sums):
        ns = s + w
        # Find where ns would be in sorted_sums
        j = bisect_left(sorted_sums, ns)
        if j < len(sorted_sums) and sorted_sums[j] == ns:
            return False
    return True

def _build_sorted_sums(weights):
    """Build sorted list of all subset sums for given weights."""
    sums = [0]
    for w in weights:
        new_sums = [s + w for s in sums]
        sums = sorted(sums + new_sums)
    return sums

def _greedy_add(sorted_sums, current_set, available, time_limit):
    """Greedily add elements that don't cause collisions."""
    for x in available:
        if x in current_set:
            continue
        w = _WEIGHTS[x]
        if _can_add(sorted_sums, w):
            current_set.add(x)
            # Rebuild sorted sums
            new_sums = [s + w for s in sorted_sums]
            sorted_sums = sorted(sorted_sums + new_sums)
        if time.time() > time_limit:
            break
    return sorted_sums, current_set

def _local_search(sorted_sums, current_set, time_limit):
    """Try swap moves to improve the set."""
    best_set = set(current_set)
    best_size = len(current_set)
    best_sums = sorted_sums
    
    for _ in range(100):  # Limited iterations
        if time.time() > time_limit:
            break
        
        improved = False
        # Try removing one element and adding another
        for x in list(best_set):
            for y in range(1, 23):
                if y in best_set:
                    continue
                
                # Try swap: remove x, add y
                new_set = set(best_set)
                new_set.remove(x)
                new_set.add(y)
                
                if len(new_set) != len(best_set):
                    continue
                
                # Build sums for new set
                weights = [_WEIGHTS[z] for z in new_set]
                try:
                    new_sums = _build_sorted_sums(weights)
                    if len(new_sums) == len(best_sums):  # All sums distinct
                        # Check if this is better (we just want to maintain size, but maybe find better structure)
                        # For now, accept if no collision
                        if not improved:
                            best_set = new_set
                            best_sums = new_sums
                            improved = True
                except:
                    pass
        
        if not improved:
            break
    
    return best_sums, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    all_elements = list(range(1, 23))
    sorted_elements = sorted(all_elements, key=lambda x: -_WEIGHTS[x])
    
    best_set = set()
    best_size = 0
    best_sums = [0]
    
    # Multiple greedy restarts with different random orders
    for trial in range(30):
        if time.time() > deadline:
            break
        
        random.shuffle(sorted_elements)
        current_set = set()
        current_sums = [0]
        
        for x in sorted_elements:
            if time.time() > deadline:
                break
            w = _WEIGHTS[x]
            if _can_add(current_sums, w):
                current_set.add(x)
                new_sums = [s + w for s in current_sums]
                current_sums = sorted(current_sums + new_sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            best_sums = current_sums
    
    if best_size == 0:
        return [1]
    
    # Local search to try to improve
    if time.time() < deadline - 10:  # Need 10s buffer
        best_sums, best_set = _local_search(best_sums, best_set, deadline - 5)
        best_size = len(best_set)
    
    # Backtracking search from the best greedy solution
    def backtrack(current_sums, current_mask, current_size):
        nonlocal best_set, best_size, best_sums
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(i for i in range(1, 23) if (current_mask >> (i-1)) & 1)
            best_sums = current_sums
        
        # Pruning: can't beat best_size even with all remaining
        remaining = sum(1 for i in range(1, 23) if not (current_mask >> (i-1)) & 1)
        if current_size + remaining <= best_size:
            return
        
        if time.time() > deadline:
            return
        
        # Try adding each available element, prioritizing larger weights
        for x in sorted_elements:
            bit = 1 << (x - 1)
            if current_mask & bit:
                continue
            
            w = _WEIGHTS[x]
            if _can_add(current_sums, w):
                new_sums = [s + w for s in current_sums]
                new_sums = sorted(current_sums + new_sums)
                backtrack(new_sums, current_mask | bit, current_size + 1)
                
                if time.time() > deadline:
                    return
    
    # Start backtracking from empty set to explore globally
    if time.time() < deadline - 2:
        backtrack([0], 0, 0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
