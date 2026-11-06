"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Fast Sidon check using bitset-like approach for small sets
    def is_valid(arr):
        """
        Checks if arr is a valid Sidon set covering 1..30.
        Returns (is_valid, diffs_set, diameter).
        """
        if len(arr) > 32:
            return False, None, None
        if 0 not in arr:
            return False, None, None
        if max(arr) > 10**6:
            return False, None, None
            
        n = len(arr)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False, None, None
                diffs.add(d)
        
        # Check coverage 1..30
        for d in range(1, 31):
            if d not in diffs:
                return False, None, None
        return True, diffs, max(arr) - min(arr)

    def fast_check_sidon(arr):
        """Fast check for Sidon property only."""
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def greedy_construct(target_max=100, randomize=True):
        """
        Greedy construction to get a valid starting set.
        """
        # Strategy: Build set element by element, preferring elements that cover missing small differences
        best_set = None
        best_diam = float('inf')
        
        for restart in range(200):
            if time.time() > deadline - 15:
                break
                
            current = [0]
            diffs = set()
            
            # Determine target elements
            if randomize:
                # Randomized greedy
                candidates = list(range(1, target_max + 1))
                random.shuffle(candidates)
            else:
                # Deterministic greedy
                candidates = list(range(1, target_max + 1))
            
            for c in candidates:
                # Check if adding c maintains Sidon
                new_diffs = set()
                ok = True
                for x in current:
                    d = c - x
                    if d <= 0:
                        continue
                    if d in diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                
                if not ok:
                    continue
                
                # Add it
                current.append(c)
                diffs.update(new_diffs)
                current.sort()
                
                # Check if we've covered 1..30
                if all(d in diffs for d in range(1, 31)):
                    diam = max(current) - min(current)
                    if diam < best_diam:
                        best_diam = diam
                        best_set = current[:]
                    break
            
            if best_set is not None and best_diam < 60:
                break

        if best_set is not None:
            return best_set
        
        # Fallback: Deterministic greedy
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                if x > y:
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
            if valid:
                for y in a:
                    if x > y:
                        diffs.add(x - y)
                a.append(x)
            x += 1
        return a

    def local_search(initial_set, time_limit=None):
        """
        Hill climbing / Simulated Annealing to minimize diameter.
        """
        if time_limit is None:
            time_limit = deadline - 5
        
        current = list(initial_set)
        current.sort()
        valid, _, cur_diam = is_valid(current)
        if not valid:
            return initial_set
        
        best_set = current[:]
        best_diam = cur_diam
        T = 50.0
        T_min = 0.5
        cooling = 0.9999
        step_count = 0
        
        while time.time() < time_limit:
            step_count += 1
            if T < T_min:
                T = T_min
            
            # Generate neighbor
            neighbor = current[:]
            move_type = random.random()
            
            if move_type < 0.5:
                # Swap: Replace a non-zero element
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    max_val = max(neighbor)
                    # Bias towards smaller values to reduce diameter
                    if random.random() < 0.7:
                        new_val = random.randint(1, max_val)
                    else:
                        new_val = random.randint(1, max_val + 20)
                    
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        neighbor = current[:]
            
            elif move_type < 0.7 and len(neighbor) > 1:
                # Remove a non-zero element
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    neighbor.pop(idx)
                    neighbor.sort()
            
            elif move_type < 0.85 and len(neighbor) < 32:
                # Add a new element
                max_val = max(neighbor)
                if random.random() < 0.5:
                    new_val = random.randint(1, max_val)
                else:
                    new_val = random.randint(1, max_val + 10)
                
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
            
            else:
                # Perturbation: change multiple elements
                num_changes = random.randint(1, 3)
                for _ in range(num_changes):
                    idxs = [i for i, v in enumerate(neighbor) if v != 0]
                    if idxs:
                        idx = random.choice(idxs)
                        max_val = max(neighbor)
                        new_val = random.randint(1, max_val + 5)
                        if new_val not in neighbor:
                            neighbor[idx] = new_val
                            neighbor.sort()
            
            # Evaluate
            if len(neighbor) > 32 or 0 not in neighbor:
                continue
            if max(neighbor) > 10**6:
                continue
            
            # Fast Sidon check first
            if not fast_check_sidon(neighbor):
                continue
            
            # Full validity check
            v, diffs, new_diam = is_valid(neighbor)
            if not v:
                continue
            
            delta = new_diam - best_diam
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > 0 and random.random() < pow(2.71828, -delta / T)):
                current = neighbor
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_set = neighbor[:]
            
            T *= cooling
            
        return best_set

    def backtracking_solve(D, max_depth=30):
        """
        Try to find a valid Sidon set with max element <= D.
        Returns the set if found, None otherwise.
        Uses backtracking with pruning.
        """
        if time.time() > deadline - 3:
            return None
            
        elements = [0]
        diffs = set()
        
        def backtrack(remaining):
            if time.time() > deadline - 3:
                return None
            
            if all(d in diffs for d in range(1, 31)):
                return elements[:]
            
            if not remaining:
                return None
            
            # Try each remaining element
            for idx, c in enumerate(remaining):
                new_diffs = set()
                ok = True
                for x in elements:
                    d = c - x
                    if d <= 0:
                        continue
                    if d in diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                
                if not ok:
                    continue
                
                elements.append(c)
                diffs.update(new_diffs)
                
                new_remaining = remaining[:idx] + remaining[idx+1:]
                result = backtrack(new_remaining)
                if result is not None:
                    return result
                
                elements.pop()
                diffs.difference_update(new_diffs)
            
            return None
        
        # Generate candidate elements in range [1, D]
        candidates = list(range(1, D + 1))
        random.shuffle(candidates)
        
        # Limit search depth for speed
        search_candidates = candidates[:min(len(candidates), max_depth)]
        
        return backtrack(search_candidates)

    # Main Loop
    best_overall = None
    best_overall_diam = float('inf')
    
    # Phase 1: Get initial set via greedy
    init_set = greedy_construct(target_max=80, randomize=True)
    
    if init_set is not None:
        result = local_search(init_set, time_limit=deadline - 30)
        v, _, diam = is_valid(result)
        if v:
            best_overall = result
            best_overall_diam = diam
    
    # Phase 2: Try backtracking for smaller diameters
    if best_overall is not None:
        current_best_diam = best_overall_diam
        
        # Try to find solutions with smaller diameter
        for D in range(current_best_diam - 1, max(30, current_best_diam - 20), -1):
            if time.time() > deadline - 10:
                break
                
            # Try multiple random orderings
            found = None
            for attempt in range(10):
                if time.time() > deadline - 10:
                    break
                res = backtracking_solve(D, max_depth=25)
                if res is not None:
                    found = res
                    break
            
            if found is not None:
                v, _, new_diam = is_valid(found)
                if v and new_diam < best_overall_diam:
                    best_overall = found
                    best_overall_diam = new_diam
                    # Continue trying smaller
                    continue
            else:
                # If we failed to find a solution for D, likely D is too small
                # But it could be due to search limits, so be cautious
                pass
    
    # Phase 3: More local search if time permits
    if time.time() < deadline - 10 and best_overall is not None:
        result = local_search(best_overall, time_limit=deadline - 5)
        v, _, diam = is_valid(result)
        if v and diam < best_overall_diam:
            best_overall = result
            best_overall_diam = diam
    
    if best_overall is None:
        # Emergency fallback
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                if x > y:
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
            if valid:
                for y in a:
                    if x > y:
                        diffs.add(x - y)
                a.append(x)
            x += 1
        return a

    return best_overall
# EVOLVE-BLOCK-END
