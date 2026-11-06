"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_sums_np(chosen):
    """Count all pairwise sums a+b where a<=b from chosen using numpy."""
    arr = np.array(chosen, dtype=np.int32)
    n = len(arr)
    if n == 0:
        return np.zeros(2001, dtype=np.int32)
    # Compute all pairwise sums (upper triangle including diagonal)
    i_idx, j_idx = np.triu_indices(n, 0)
    sums = arr[i_idx] + arr[j_idx]
    counts = np.zeros(2001, dtype=np.int32)
    np.add.at(counts, sums, 1)
    return counts

def _can_add_np(counts, x, chosen_set):
    """Check if x can be added without violating B2[2]."""
    # Check sums with existing elements
    for y in chosen_set:
        s = x + y
        if s <= 2000 and counts[s] >= 2:
            return False
    # Check 2*x
    if 2 * x <= 2000 and counts[2 * x] >= 2:
        return False
    return True

def _greedy_construct(order, counts=None):
    """Greedy construction following a specific order of candidates."""
    if counts is None:
        counts = np.zeros(2001, dtype=np.int32)
    else:
        counts = counts.copy()
    chosen = []
    chosen_set = set()
    for x in order:
        if _can_add_np(counts, x, chosen_set):
            chosen.append(x)
            chosen_set.add(x)
            # Update counts
            for y in chosen[:-1]:
                s = x + y
                counts[s] += 1
            counts[2 * x] += 1
    return chosen

def _greedy_density_aware():
    """Greedy that prioritizes elements with fewer potential conflicts."""
    counts = np.zeros(2001, dtype=np.int32)
    chosen = []
    chosen_set = set()
    candidates = list(range(1, 1001))
    
    while candidates:
        # For each candidate, compute a "cost" = number of sums that would hit count >= 2
        best_x = None
        best_cost = float('inf')
        best_idx = -1
        
        # Sample a subset if too many candidates
        sample_size = min(len(candidates), 200)
        sample_indices = random.sample(range(len(candidates)), sample_size)
        
        for idx in sample_indices:
            x = candidates[idx]
            cost = 0
            for y in chosen_set:
                s = x + y
                if s <= 2000 and counts[s] >= 2:
                    cost += 1
            s2 = 2 * x
            if s2 <= 2000 and counts[s2] >= 2:
                cost += 1
            
            if cost < best_cost:
                best_cost = cost
                best_x = x
                best_idx = idx
            
            if cost == 0:
                break
        
        if best_x is None or best_cost > 0:
            # No zero-cost option, try all remaining
            for idx in range(len(candidates)):
                x = candidates[idx]
                cost = 0
                for y in chosen_set:
                    s = x + y
                    if s <= 2000 and counts[s] >= 2:
                        cost += 1
                s2 = 2 * x
                if s2 <= 2000 and counts[s2] >= 2:
                    cost += 1
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                    best_idx = idx
                if cost == 0:
                    break
        
        if best_x is None:
            break
        
        # Add the best candidate
        chosen.append(best_x)
        chosen_set.add(best_x)
        for y in chosen[:-1]:
            s = best_x + y
            counts[s] += 1
        counts[2 * best_x] += 1
        candidates.pop(best_idx)
        
        if time.time() > 120:
            break
    
    return chosen

def _local_improve_fast(chosen, max_iters=200, deadline=130):
    """Efficient local improvement by trying swaps."""
    chosen = sorted(chosen)
    counts = _count_sums_np(chosen)
    chosen_set = set(chosen)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > deadline:
            break
        
        # Try removing each element and adding a different one
        for i in range(len(chosen)):
            x_remove = chosen[i]
            # Temporarily remove x_remove
            temp_counts = counts.copy()
            for y in chosen:
                if y == x_remove:
                    continue
                s = x_remove + y
                temp_counts[s] -= 1
            temp_counts[2 * x_remove] -= 1
            
            # Try adding each candidate not in temp_chosen
            temp_chosen_set = chosen_set - {x_remove}
            best_add = None
            best_score = -1
            
            # Sample candidates to speed up
            candidates = [x for x in range(1, 1001) if x not in temp_chosen_set]
            random.shuffle(candidates)
            sample = candidates[:100]
            
            for x_add in sample:
                # Check if valid
                valid = True
                for y in temp_chosen_set:
                    s = x_add + y
                    if temp_counts[s] >= 2:
                        valid = False
                        break
                if valid and temp_counts[2 * x_add] >= 2:
                    valid = False
                
                if valid:
                    # Score: prefer additions that create fewer future conflicts
                    score = 0
                    for y in temp_chosen_set:
                        s = x_add + y
                        if temp_counts[s] == 1:
                            score += 1
                    if temp_counts[2 * x_add] == 1:
                        score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_add = x_add
            
            if best_add is not None:
                # Perform the swap
                chosen[i] = best_add
                chosen_set.remove(x_remove)
                chosen_set.add(best_add)
                chosen = sorted(chosen)
                counts = _count_sums_np(chosen)
                improved = True
                break
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Natural ascending greedy
    order1 = list(range(1, 1001))
    set1 = _greedy_construct(order1)
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1
    
    if time.time() > deadline - 60:
        return best_set
    
    # Strategy 2: Descending greedy
    order2 = list(range(1000, 0, -1))
    set2 = _greedy_construct(order2)
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2
    
    if time.time() > deadline - 50:
        return best_set
    
    # Strategy 3: Density-aware greedy with multiple random seeds
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        s = _greedy_density_aware()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    if time.time() > deadline - 25:
        return best_set
    
    # Local improvement on the best set found
    improved = _local_improve_fast(best_set, max_iters=100, deadline=deadline - 5)
    if len(improved) > len(best_set):
        best_set = improved
    
    return best_set
# EVOLVE-BLOCK-END
