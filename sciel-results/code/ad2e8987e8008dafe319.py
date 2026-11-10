"""Improved program for two Sidon sets with disjoint difference sets."""
import time
import random
from itertools import combinations, combinations_with_replacement

def _is_sidon(arr):
    """Check if arr is a Sidon set (all pair sums a+b with a<=b are unique)."""
    n = len(arr)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(arr):
    """Return set of nonzero differences |b-a| for a != b in arr."""
    d = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d.add(abs(arr[j] - arr[i]))
    return d

def _greedy_construct(order, deadline):
    """Greedy construction given an order of candidates."""
    sets = [[], []]
    diff_sets = [set(), set()]
    sum_sets = [set(), set()]
    
    for x in order:
        for side in (0, 1):
            if time.time() > deadline:
                return sets
            
            # Check Sidon condition for this side
            candidate = sets[side] + [x]
            valid_sidon = True
            for i in range(len(candidate)):
                for j in range(i, len(candidate)):
                    s = candidate[i] + candidate[j]
                    if s in sum_sets[side]:
                        valid_sidon = False
                        break
                if not valid_sidon:
                    break
            
            if not valid_sidon:
                continue
            
            # Check difference disjointness
            candidate_diffs = set()
            for i in range(len(candidate)):
                for j in range(i + 1, len(candidate)):
                    d = abs(candidate[j] - candidate[i])
                    candidate_diffs.add(d)
            
            if candidate_diffs & diff_sets[1 - side]:
                continue
            
            # Accept
            sets[side].append(x)
            for i in range(len(candidate)):
                for j in range(i, len(candidate)):
                    sum_sets[side].add(candidate[i] + candidate[j])
            diff_sets[side] = candidate_diffs
            break
    
    return sets

def _local_improve(sets, deadline):
    """Try to improve by swapping or adding elements."""
    best = [list(sets[0]), list(sets[1])]
    best_size = len(best[0]) + len(best[1])
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        # Try adding random elements
        for side in (0, 1):
            for _ in range(50):
                x = random.randint(1, 256)
                if x in best[side]:
                    continue
                
                candidate = best[side] + [x]
                if not _is_sidon(candidate):
                    continue
                
                new_diffs = _diffs(candidate)
                other_diffs = _diffs(best[1 - side])
                if new_diffs & other_diffs:
                    continue
                
                best[side].append(x)
                if len(best[0]) + len(best[1]) > best_size:
                    best_size = len(best[0]) + len(best[1])
                    improved = True
                break
        
        # Try swapping elements
        for side in (0, 1):
            if not best[side]:
                continue
            for _ in range(20):
                idx = random.randint(0, len(best[side]) - 1)
                old_x = best[side][idx]
                new_x = random.randint(1, 256)
                if new_x == old_x or new_x in best[side]:
                    continue
                
                candidate = best[side][:idx] + [new_x] + best[side][idx + 1:]
                if not _is_sidon(candidate):
                    continue
                
                new_diffs = _diffs(candidate)
                other_diffs = _diffs(best[1 - side])
                if new_diffs & other_diffs:
                    continue
                
                best[side] = candidate
                if len(best[0]) + len(best[1]) > best_size:
                    best_size = len(best[0]) + len(best[1])
                    improved = True
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_result = [[], []]
    best_size = 0
    
    # Multiple randomized restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Shuffle candidate order
        order = list(range(1, 257))
        random.shuffle(order)
        
        # Greedy construction
        result = _greedy_construct(order, deadline)
        
        # Local improvement
        if time.time() < deadline - 5.0:
            result = _local_improve(result, deadline)
        
        current_size = len(result[0]) + len(result[1])
        if current_size > best_size:
            best_size = current_size
            best_result = result
        
        # Early exit if we hit the theoretical upper bound
        if best_size >= 128:
            break
    
    # Ensure nonempty
    if not best_result[0]:
        best_result[0] = [1]
    if not best_result[1]:
        best_result[1] = [2]
    
    return best_result
