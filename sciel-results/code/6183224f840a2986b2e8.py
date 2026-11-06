import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin before 120s limit
    N = 284
    MANDATORY = (1, 2, 5, 11)
    
    best_size = len(MANDATORY)
    best_set = set(MANDATORY)
    
    def get_sums(arr):
        """Get all pairwise sums a+b for a<=b in arr."""
        s = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                s.add(arr[i] + arr[j])
        return s
    
    def is_valid_addition(arr, x, sums):
        """Check if x can be added to arr without creating duplicate sums."""
        if x < 1 or x > N:
            return False
        for y in arr:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def greedy_fill(arr, sums, candidates, rng):
        """Fill arr greedily by trying candidates in random order."""
        nonlocal best_size, best_set
        current = list(arr)
        current_sums = set(sums)
        
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if is_valid_addition(current, x, current_sums):
                current.append(x)
                current_sums.update({x + y for y in current[:-1]} | {2 * x})
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
        
        return current
    
    def local_search(arr, sums, rng, max_iters=50):
        """Try to improve by swapping elements."""
        nonlocal best_size, best_set
        current = list(arr)
        current_sums = set(sums)
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try swapping a random element with a candidate
            if not current:
                break
            
            idx = rng.randint(0, len(current) - 1)
            old_val = current[idx]
            
            # Try to find a replacement
            candidates = [x for x in range(1, N + 1) if x not in current]
            rng.shuffle(candidates)
            
            found = False
            for new_val in candidates[:20]:  # Limit search
                if is_valid_addition([v for v in current if v != old_val], new_val, 
                                     set(s for s in current_sums if old_val not in s)):
                    # More careful check needed
                    pass
            
            # Simpler: try removing and re-adding
            current_sums = get_sums(current)
            for new_val in candidates[:30]:
                temp = [v for v in current if v != old_val]
                if is_valid_addition(temp, new_val, get_sums(temp)):
                    current[idx] = new_val
                    current.sort()
                    current_sums = get_sums(current)
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    found = True
                    break
            
            if not found:
                break
        
        return current
    
    def backtracking_search(arr, sums, remaining, target_size, rng, depth=0):
        """Backtracking with forward checking."""
        nonlocal best_size, best_set
        if time.time() - start_time > DEADLINE:
            return None
        
        if len(arr) >= target_size:
            if len(arr) > best_size:
                best_size = len(arr)
                best_set = set(arr)
            return arr
        
        if depth > 30:  # Limit depth
            return None
        
        # MRV heuristic: try candidates that leave most flexibility
        candidates = [x for x in remaining if is_valid_addition(arr, x, sums)]
        if not candidates:
            return None
        
        # Sort by how many future options they preserve (approximate)
        def flexibility(x):
            count = 0
            for y in candidates:
                if y != x and is_valid_addition(arr + [x], y, sums | {x + y for y in arr} | {2*x}):
                    count += 1
            return count
        
        # Use a simpler heuristic: try values that are "spread out"
        rng.shuffle(candidates)
        
        for x in candidates[:15]:  # Limit branching factor
            new_arr = arr + [x]
            new_sums = sums | {x + y for y in arr} | {2 * x}
            
            result = backtracking_search(new_arr, new_sums, 
                                         [y for y in remaining if y != x], 
                                         target_size, rng, depth + 1)
            if result is not None and len(result) > best_size:
                best_size = len(result)
                best_set = set(result)
                return result
        
        return None
    
    # Main search loop with multiple strategies
    rng = random.Random(42)
    
    # Strategy 1: Greedy with random ordering
    for iteration in range(1000):
        if time.time() - start_time > DEADLINE:
            break
        
        candidates = list(range(1, N + 1))
        for m in MANDATORY:
            candidates.remove(m)
        rng.shuffle(candidates)
        
        initial = list(MANDATORY)
        initial_sums = get_sums(initial)
        
        greedy_fill(initial, initial_sums, candidates, rng)
    
    # Strategy 2: Greedy + local search
    for iteration in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        candidates = list(range(1, N + 1))
        for m in MANDATORY:
            candidates.remove(m)
        rng.shuffle(candidates)
        
        initial = list(MANDATORY)
        initial_sums = get_sums(initial)
        
        arr = greedy_fill(initial, initial_sums, candidates, rng)
        arr_sums = get_sums(arr)
        local_search(arr, arr_sums, rng, max_iters=20)
    
    # Strategy 3: Targeted backtracking for larger sets
    for target in range(best_size + 1, 21):
        if time.time() - start_time > DEADLINE:
            break
        
        candidates = list(range(1, N + 1))
        for m in MANDATORY:
            candidates.remove(m)
        
        initial = list(MANDATORY)
        initial_sums = get_sums(initial)
        
        result = backtracking_search(initial, initial_sums, candidates, target, rng)
        if result is not None:
            break
    
    return sorted(best_set)
