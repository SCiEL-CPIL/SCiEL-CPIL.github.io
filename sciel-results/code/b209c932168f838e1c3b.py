import time
import random

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

def _greedy_construct_fast(max_size=60, deadline=None, rng=None):
    """Fast greedy construction with a given ordering."""
    if deadline is None:
        deadline = time.time() + 5
    if rng is None:
        rng = random.Random()
    
    current = []
    current_set = set()
    
    # Generate an ordering
    candidates = list(range(1, 257))
    if rng:
        rng.shuffle(candidates)
    else:
        # Natural order
        pass
        
    for candidate in candidates:
        if time.time() > deadline:
            break
        if candidate in current_set:
            continue
        if len(current) >= max_size:
            break
            
        # Check if adding candidate is valid
        # We can optimize this by pre-computing bitsets for current set
        # But _is_valid_bitset on the whole list is O(N^2 * BitLen). 
        # For N=30, BitLen=256*30 ~ 7680. 
        # 30*30*7680 ~ 7M ops. Very fast in Python if bit ops are C-level.
        test_set = current + [candidate]
        if _is_valid_bitset(test_set):
            current.append(candidate)
            current_set.add(candidate)
            
    return current

def _try_ap(k):
    """Try to find a valid AP of size k."""
    # For AP a, a+d, ..., a+(k-1)d
    # We found k=4 fails for d=10. 
    # Let's just test a few small APs to see if any work.
    # Actually, if k=15 works, we need to find the parameters.
    # But greedy is more flexible.
    pass

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    best = [1]
    best_size = 1
    
    # 1. Try natural order greedy
    sol = _greedy_construct_fast(max_size=60, deadline=deadline, rng=None)
    if len(sol) > best_size:
        best = sol
        best_size = len(sol)
        
    # 2. Try multiple random orderings
    # Each run takes ~0.1s? 
    # We can run many.
    start_time = time.time()
    attempts = 0
    while time.time() < deadline - 5:
        rng = random.Random()
        sol = _greedy_construct_fast(max_size=60, deadline=deadline, rng=rng)
        if len(sol) > best_size:
            best = sol
            best_size = len(sol)
            # If we hit 60, we can stop early? No, max is 60.
            if best_size == 60:
                break
        attempts += 1
        
    # 3. Local improvement on the best solution found
    if time.time() < deadline - 2 and best_size < 60:
        current = list(best)
        current_set = set(current)
        improved = False
        # Try adding any remaining number
        for candidate in range(1, 257):
            if time.time() > deadline:
                break
            if candidate in current_set:
                continue
            test_set = current + [candidate]
            if _is_valid_bitset(test_set):
                current.append(candidate)
                current_set.add(candidate)
                improved = True
                if len(current) > best_size:
                    best = current
                    best_size = len(current)
                    
        # Try swaps if still not 60
        if not improved and time.time() < deadline - 1:
            for _ in range(100):
                if time.time() > deadline:
                    break
                if not current:
                    break
                idx = random.randrange(len(current))
                removed = current[idx]
                base = [x for x in current if x != removed]
                base_set = set(base)
                for candidate in range(1, 257):
                    if candidate in base_set:
                        continue
                    test_set = base + [candidate]
                    if _is_valid_bitset(test_set):
                        current = test_set
                        current_set = set(test_set)
                        if len(current) > best_size:
                            best = current
                            best_size = len(current)
                        break
                        
    return list(best)
