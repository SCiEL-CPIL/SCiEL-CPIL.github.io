"""Integer sum-product extremal sets - minimize max(|A+A|, |A*A|)"""
import time
import random
import math
import numpy as np

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    # Sumset with equal summands allowed (i <= j)
    i_idx, j_idx = np.triu_indices(n, k=0)
    sums = A_arr[i_idx] + A_arr[j_idx]
    n_sums = len(np.unique(sums))
    
    # Product set with equal factors allowed (i <= j)
    products = A_arr[i_idx] * A_arr[j_idx]
    n_products = len(np.unique(products))
    
    return max(n_sums, n_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _generate_ap_union(num_aps, lengths, diffs, rng):
    """
    Generate a set that is a union of short arithmetic progressions.
    This exploits additive structure to keep |A+A| small.
    """
    A = set()
    base_offset = 0
    for i in range(num_aps):
        start = base_offset + rng.randint(1, 100)
        d = diffs[i]
        l = lengths[i]
        for j in range(l):
            val = start + j * d
            if 1 <= val <= 10000 and val not in A:
                A.add(val)
        # Increase base offset to avoid overlap with previous APs
        base_offset += l * d + rng.randint(10, 50)
        
    # If we still need more elements, add random ones
    while len(A) < 20:
        v = rng.randint(1, 10000)
        if v not in A:
            A.add(v)
            
    # If we have too many (shouldn't happen if we control lengths, but just in case)
    if len(A) > 20:
        A_list = list(A)
        A = set(random.sample(A_list, 20))
        
    return list(A)

def _generate_gp(ratio, start_val):
    """Generate a geometric progression set, filling gaps if needed."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= 10000:
        A.add(current)
        current = int(current * ratio)
        if current <= start_val:
            break
    
    if len(A) < 20:
        # Fill remaining with random small integers to maintain distinctness
        candidates = list(range(1, 10001))
        random.shuffle(candidates)
        for v in candidates:
            if len(A) >= 20:
                break
            if v not in A:
                A.add(v)
    return list(A)

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.99
    min_temperature = 0.01
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        found_improvement = False
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Random delta - biased toward smaller changes for refinement
            if rng.random() < 0.4:
                new_val = old_val + rng.randint(-20, 20)
            elif rng.random() < 0.7:
                new_val = old_val + rng.randint(-100, 100)
            else:
                new_val = old_val + rng.randint(-500, 500)
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                found_improvement = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _multi_start_local_search(time_budget):
    """Run multiple local searches from different starting points."""
    rng = random.Random()
    best_A = None
    best_obj = float('inf')
    
    start_time = time.time()
    
    # 1. Try unions of arithmetic progressions (Additive focus)
    # Configurations: (num_aps, [lengths], [diffs])
    ap_configs = [
        (2, [10, 10], [1, 2]),
        (2, [10, 10], [2, 3]),
        (2, [8, 12], [1, 3]),
        (3, [7, 7, 6], [1, 2, 3]),
        (3, [6, 7, 7], [2, 3, 5]),
        (4, [5, 5, 5, 5], [1, 2, 3, 4]),
    ]
    
    for config in ap_configs:
        if time.time() - start_time > time_budget:
            break
        num_aps, lengths, diffs = config
        for _ in range(5): # 5 random restarts per config
            if time.time() - start_time > time_budget:
                break
            A = _generate_ap_union(num_aps, lengths, diffs, rng)
            if not _is_valid(A):
                continue
            remaining = time_budget - (time.time() - start_time)
            if remaining < 1:
                break
            A_sorted, obj = _local_search(A, remaining, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted

    # 2. Try geometric progressions (Multiplicative focus)
    for ratio in [1.2, 1.5, 2.0]:
        for start_val in [1, 2, 3]:
            if time.time() - start_time > time_budget:
                break
            A = _generate_gp(ratio, start_val)
            if not _is_valid(A):
                continue
            remaining = time_budget - (time.time() - start_time)
            if remaining < 1:
                break
            A_sorted, obj = _local_search(A, remaining, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted

    # 3. Try random starts
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2:
        max_starts = min(20, int(remaining / 2))
        for _ in range(max_starts):
            if time.time() - start_time > time_budget:
                break
            A = random.sample(range(1, 10001), 20)
            if not _is_valid(A):
                continue
            rem = time_budget - (time.time() - start_time)
            if rem < 1:
                break
            A_sorted, obj = _local_search(A, rem, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = 85  # leave 5s safety margin for 90s limit
    
    try:
        best_A, best_obj = _multi_start_local_search(time_budget)
    except Exception:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    # Final refinement with whatever time is left
    remaining = time_budget - (time.time() - start_time)
    if remaining > 1:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, random.Random())
            if refined_obj < best_obj:
                best_A = refined_A
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
