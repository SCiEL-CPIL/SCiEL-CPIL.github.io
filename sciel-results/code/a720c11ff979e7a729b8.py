def construct():
    import time
    import random
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def get_subarray_products(v):
        """Get all subarray products with early exit on duplicate."""
        ps = set()
        n = len(v)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= v[j]
                if p in ps:
                    return None  # Invalid - duplicate found
                ps.add(p)
        return ps
    
    def is_valid_fast(v):
        """Check if sequence has all distinct subarray products."""
        return get_subarray_products(v) is not None
    
    def try_add_element(current_vals, x):
        """Try to add x to current_vals (sorted) and check validity."""
        if x in current_vals:
            return None
        candidate = sorted(current_vals + [x])
        if is_valid_fast(candidate):
            return candidate
        return None
    
    # Phase 1: Multiple greedy initializations
    best_vals = []
    best_len = 0
    
    # Strategy 1: Simple ascending greedy
    vals = []
    for x in range(2, 129):
        if try_add_element(vals, x) is not None:
            vals.append(x)
    if len(vals) > best_len:
        best_vals = list(vals)
        best_len = len(vals)
    
    # Strategy 2: Prioritize primes first, then composites
    primes = [x for x in range(2, 129) if all(x % d != 0 for d in range(2, int(x**0.5)+1) if d*d <= x)]
    composites = [x for x in range(2, 129) if x not in primes]
    vals = []
    for x in primes:
        if try_add_element(vals, x) is not None:
            vals.append(x)
    for x in composites:
        if try_add_element(vals, x) is not None:
            vals.append(x)
    if len(vals) > best_len:
        best_vals = list(vals)
        best_len = len(vals)
    
    # Strategy 3: Descending greedy (sometimes helps)
    vals = []
    for x in range(128, 1, -1):
        if try_add_element(vals, x) is not None:
            vals.append(x)
    vals.sort()
    if len(vals) > best_len:
        best_vals = list(vals)
        best_len = len(vals)
    
    # Phase 2: Local search with cycle prevention
    current = list(best_vals)
    visited = {tuple(current)}
    no_add_streak = 0
    max_no_add = 50  # Limit consecutive swaps without addition
    
    while time.time() < deadline:
        improved = False
        
        # Try to add any missing number
        for x in range(2, 129):
            if x not in current:
                candidate = try_add_element(current, x)
                if candidate is not None:
                    current = candidate
                    improved = True
                    no_add_streak = 0
                    visited.add(tuple(current))
                    break
        
        if improved:
            continue
        
        no_add_streak += 1
        if no_add_streak > max_no_add:
            break
        
        # Try swaps: replace one element with a missing one
        swapped = False
        n = len(current)
        for i in range(n):
            if time.time() >= deadline:
                break
            a = current[i]
            for b in range(2, 129):
                if b == a or b in current:
                    continue
                candidate = current[:]
                candidate[i] = b
                candidate.sort()
                key = tuple(candidate)
                if key in visited:
                    continue
                if is_valid_fast(candidate):
                    current = candidate
                    visited.add(key)
                    swapped = True
                    break
            if swapped or time.time() >= deadline:
                break
        
        if not swapped:
            # Try 2-swap moves: replace two elements with two missing ones
            two_swapped = False
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i+1, n):
                    if time.time() >= deadline:
                        break
                    a, b = current[i], current[j]
                    for c in range(2, 129):
                        if c == a or c == b or c in current:
                            continue
                        for d in range(2, 129):
                            if d == a or d == b or d == c or d in current:
                                continue
                            if c >= d:
                                continue
                            candidate = current[:]
                            candidate[i] = c
                            candidate[j] = d
                            candidate.sort()
                            key = tuple(candidate)
                            if key in visited:
                                continue
                            if is_valid_fast(candidate):
                                current = candidate
                                visited.add(key)
                                two_swapped = True
                                break
                        if two_swapped:
                            break
                    if two_swapped:
                        break
                if two_swapped:
                    break
            
            if not two_swapped:
                break
    
    return current
