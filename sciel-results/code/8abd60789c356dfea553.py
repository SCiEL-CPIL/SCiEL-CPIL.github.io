import time
import random

def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    # Use Python integers as bitsets (arbitrary precision)
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1  # bit 0 set: sum 0 with cardinality 0
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
    # Check disjointness across different cardinalities
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if sums_by_card[j] & sums_by_card[k]:
                return False
    return True

def _validate(arr):
    if len(arr) != len(set(arr)):
        return False
    if any(v < 1 or v > 256 for v in arr):
        return False
    return _is_valid_bitset(arr)

def _ap_solution(k):
    """Return AP of size k if it fits in 1..256 and is valid, else None."""
    if k == 1:
        return [1]
    if k == 2:
        return [1, 2]
    # For AP a, a+1, ..., a+k-1
    # Need a > max_{1<=j<=k-1} j*(k-j-1)
    max_val = 0
    for j in range(1, k):
        v = j * (k - j - 1)
        if v > max_val:
            max_val = v
    required_a = max_val + 1
    if required_a <= 256 - k + 1:
        set_vals = list(range(required_a, required_a + k))
        if _validate(set_vals):
            return set_vals
    return None

def _local_search(start_set, deadline, max_iters=5000):
    """Try to improve a set via local search."""
    current = list(start_set)
    current_size = len(current)
    best = list(current)
    best_size = current_size
    
    current_set = set(current)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if current_size >= 60:
            break
        
        # Try to add a random element
        candidates = [x for x in range(1, 257) if x not in current_set]
        if not candidates:
            break
            
        # Try a few random candidates
        tried = 0
        for _ in range(min(20, len(candidates))):
            if time.time() > deadline:
                break
            cand = random.choice(candidates)
            test_set = current + [cand]
            if _validate(test_set):
                current = test_set
                current_set.add(cand)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best = list(current)
                break
            tried += 1
        
        # Occasionally try to swap to free up space
        if current_size < best_size and random.random() < 0.1:
            # Try removing a large element and adding a smaller one
            if current:
                remove_idx = random.randint(0, len(current) - 1)
                removed = current[remove_idx]
                new_set = current[:remove_idx] + current[remove_idx+1:]
                new_set_set = set(new_set)
                # Try to add a smaller element
                small_candidates = [x for x in range(1, removed) if x not in new_set_set]
                if small_candidates:
                    cand = random.choice(small_candidates)
                    test_set = new_set + [cand]
                    if _validate(test_set):
                        current = test_set
                        current_set = set(current)
                        current_size = len(current)
                        if current_size > best_size:
                            best_size = current_size
                            best = list(current)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin for 90s limit
    
    # First, find the best AP solution
    best_ap = None
    best_ap_size = 0
    for k in range(60, 0, -1):
        if time.time() > deadline:
            break
        sol = _ap_solution(k)
        if sol is not None:
            best_ap = sol
            best_ap_size = k
            break
    
    if best_ap is None:
        best_ap = [1]
        best_ap_size = 1
    
    # Now try to find a better non-AP solution via local search
    # Start from the AP solution and try to expand
    best = list(best_ap)
    best_size = best_ap_size
    
    # Try local search from AP starting point
    if time.time() < deadline - 10:
        ls_result = _local_search(best_ap, deadline)
        if len(ls_result) > best_size:
            best = ls_result
            best_size = len(best)
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        # Start from a random valid small set
        random_set = [random.randint(1, 256)]
        random_set_set = set(random_set)
        # Try to build up
        current = list(random_set)
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            if len(current) >= best_size:
                break
            candidates = [x for x in range(1, 257) if x not in random_set_set]
            if not candidates:
                break
            added = False
            for cand in random.sample(candidates, min(10, len(candidates))):
                if time.time() > deadline - 5:
                    break
                test_set = current + [cand]
                if _validate(test_set):
                    current = test_set
                    random_set_set.add(cand)
                    added = True
                    break
            if not added:
                break
        if len(current) > best_size:
            best = current
            best_size = len(best)
    
    # Final validation
    if not _validate(best):
        return [1]
    
    return list(best)
