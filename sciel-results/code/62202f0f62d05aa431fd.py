import time

def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
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
    max_j_val = max(j * (k - 1 - j) for j in range(1, k))
    required_x_1 = max_j_val + 1
    if required_x_1 <= 256 - k + 1:
        x_1 = required_x_1
        set_vals = list(range(x_1, x_1 + k))
        if len(set(set_vals)) == k and all(1 <= v <= 256 for v in set_vals):
            if _is_valid_bitset(set_vals):
                return set_vals
    return None

def _greedy_construct(max_size=60, deadline=None):
    """Greedy construction: add elements one by one if validity is maintained."""
    if deadline is None:
        deadline = time.time() + 80
    
    best = [1]
    best_size = 1
    
    # Try adding elements in order from smallest to largest
    # Start with a base set and try to expand
    candidates = list(range(1, 257))
    
    # Try different starting points and orderings
    for start in [1, 2, 3, 5, 8, 10, 13, 16, 20]:
        if time.time() > deadline:
            break
        
        current = [start]
        for candidate in candidates:
            if time.time() > deadline:
                break
            if candidate in current:
                continue
            if len(current) >= max_size:
                break
            
            test_set = current + [candidate]
            if _is_valid_bitset(test_set):
                current = test_set
                if len(current) > best_size:
                    best_size = len(current)
                    best = list(current)
    
    return best

def construct():
    deadline = time.time() + 75
    
    # First try AP solutions (fast and known valid)
    for k in range(60, 0, -1):
        sol = _ap_solution(k)
        if sol is not None:
            ap_size = len(sol)
            # Try to beat this with greedy
            greedy_sol = _greedy_construct(max_size=60, deadline=deadline)
            if len(greedy_sol) > ap_size:
                return list(greedy_sol)
            return list(sol)
    
    # Fallback to greedy
    greedy_sol = _greedy_construct(max_size=60, deadline=deadline)
    if greedy_sol:
        return list(greedy_sol)
    
    return [1]
