# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # Precompute forbidden as a boolean array for speed
    forbidden_arr = np.zeros(m, dtype=bool)
    for f in forbidden:
        forbidden_arr[f] = True
    
    def is_valid_pair(a, b):
        d = (a - b) % m
        return not forbidden_arr[d]
    
    def is_valid_subset(subset):
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                if not is_valid_pair(subset[i], subset[j]):
                    return False
        return True
    
    def subset_size(subset):
        return len(subset)
    
    def greedy_init(order, forbidden_arr, m):
        """Greedy: process candidates in given order, add if compatible."""
        selected = []
        for candidate in order:
            ok = True
            for s in selected:
                d = (candidate - s) % m
                if forbidden_arr[d]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected
    
    def local_search(subset, forbidden_arr, m, max_iters=50000):
        """Local search: try add, remove, swap moves."""
        current = list(subset)
        current_set = set(current)
        best = list(current)
        best_size = len(current)
        
        for it in range(max_iters):
            improved = False
            
            # Try adding an element
            for cand in range(m):
                if cand in current_set:
                    continue
                ok = True
                for s in current:
                    d = (cand - s) % m
                    if forbidden_arr[d]:
                        ok = False
                        break
                if ok:
                    current.append(cand)
                    current_set.add(cand)
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing one element and adding two
            for i in range(len(current)):
                removed = current[i]
                temp = [x for j, x in enumerate(current) if j != i]
                temp_set = set(temp)
                
                # Try to add two elements
                added = []
                for cand in range(m):
                    if cand in temp_set or cand == removed:
                        continue
                    ok = True
                    for s in temp:
                        d = (cand - s) % m
                        if forbidden_arr[d]:
                            ok = False
                            break
                    if ok and len(added) < 2:
                        added.append(cand)
                        temp_set.add(cand)
                
                if len(added) >= 2:
                    new_current = temp + added
                    if len(new_current) > best_size:
                        current = new_current
                        current_set = set(current)
                        best = list(current)
                        best_size = len(current)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swap: remove one, add one different
            for i in range(len(current)):
                removed = current[i]
                temp = [x for j, x in enumerate(current) if j != i]
                temp_set = set(temp)
                
                for cand in range(m):
                    if cand in temp_set or cand == removed:
                        continue
                    ok = True
                    for s in temp:
                        d = (cand - s) % m
                        if forbidden_arr[d]:
                            ok = False
                            break
                    if ok:
                        new_current = temp + [cand]
                        if len(new_current) > best_size:
                            current = new_current
                            current_set = set(current)
                            best = list(current)
                            best_size = len(current)
                            improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    # Start with greedy from natural order
    best_subset = greedy_init(list(range(m)), forbidden_arr, m)
    
    # Try multiple random orderings
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        subset = greedy_init(order, forbidden_arr, m)
        if len(subset) > len(best_subset):
            best_subset = subset
    
    # Local search from best so far
    if time.time() < deadline - 60:
        improved = local_search(best_subset, forbidden_arr, m, max_iters=30000)
        if len(improved) > len(best_subset):
            best_subset = improved
    
    # More local search with different random restarts
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        subset = greedy_init(order, forbidden_arr, m)
        if len(subset) >= len(best_subset):
            improved = local_search(subset, forbidden_arr, m, max_iters=10000)
            if len(improved) > len(best_subset):
                best_subset = improved
    
    return np.array(best_subset, dtype=np.int64)


# EVOLVE-BLOCK-END
