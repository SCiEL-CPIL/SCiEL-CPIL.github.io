import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def check_sidon(arr):
        """
        Returns True if arr is a Sidon set.
        """
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(arr):
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(arr[j] - arr[i])
        return diffs

    def mrv_backtracking(target_diam, max_elements=32):
        """
        Attempts to find a Sidon set of size <= max_elements, containing 0,
        with max element <= target_diam, covering all diffs 1..30.
        
        Strategy:
        1. Start with [0].
        2. Identify 'required' diffs 1..30.
        3. Recursively pick the next element x.
        4. Heuristic: Pick x that covers the most 'uncovered' required diffs 
           while maintaining Sidon property.
        5. Pruning: If we run out of candidates or time, fail.
        """
        required = set(range(1, 31))
        current = [0]
        current_diffs = set()
        
        # Candidates are integers 1..target_diam
        # We will sort candidates by a heuristic score dynamically
        
        def backtrack(remaining_required):
            if time.time() > deadline - 2:
                return None
            
            if not remaining_required:
                return current[:]
            
            if len(current) >= max_elements:
                return None
                
            # Heuristic: Find the 'hardest' remaining difference to cover?
            # Or just pick the best candidate?
            # Let's pick the candidate that covers the most new required diffs.
            
            best_candidate = None
            best_new_diffs = set()
            best_score = -1
            
            # To speed up, we don't iterate all 1..target_diam every time if possible,
            # but for small target_diam (< 100), it's fine.
            # We need to check Sidon property for each candidate.
            
            # Optimization: Only consider candidates that are > max(current)
            # because we maintain sorted order.
            min_cand = max(current) + 1
            
            for x in range(min_cand, target_diam + 1):
                # Check Sidon with x
                is_ok = True
                new_diffs_for_x = set()
                for y in current:
                    d = x - y
                    if d in current_diffs or d in new_diffs_for_x:
                        is_ok = False
                        break
                    new_diffs_for_x.add(d)
                
                if not is_ok:
                    continue
                
                # Score: number of required diffs this x covers
                covered = len(new_diffs_for_x & remaining_required)
                
                # Tie-breaker: Prefer smaller x to keep diameter small? 
                # Actually, we are fixing target_diam, so any x is valid.
                # But covering more required diffs is better.
                
                if covered > best_score:
                    best_score = covered
                    best_candidate = x
                    best_new_diffs = new_diffs_for_x
            
            if best_candidate is None:
                return None
            
            # Commit
            current.append(best_candidate)
            current_diffs.update(best_new_diffs)
            new_remaining = remaining_required - best_new_diffs
            
            result = backtrack(new_remaining)
            if result is not None:
                return result
            
            # Backtrack
            current.pop()
            current_diffs.difference_update(best_new_diffs)
            
            return None

        return backtrack(required)

    def refine_with_local_search(initial_set, time_limit):
        """
        Local search to minimize diameter.
        Uses a more targeted move: swap an element with a smaller one 
        if it doesn't break Sidon/coverage.
        """
        current = list(initial_set)
        current.sort()
        best = current[:]
        best_diam = max(current)
        
        # Precompute diffs
        cur_diffs = get_diffs(current)
        
        T = 20.0
        T_min = 1.0
        cooling = 0.999
        
        while time.time() < time_limit:
            if T < T_min:
                T = T_min
            
            neighbor = current[:]
            move_type = random.random()
            
            if move_type < 0.6:
                # Swap a non-zero element
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    # Try to replace with a smaller value to reduce max
                    max_val = max(neighbor)
                    # Bias towards lower half
                    new_val = random.randint(1, max_val // 2 if max_val > 2 else 1)
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        continue
            elif move_type < 0.8:
                # Remove an element (if size > minimum required)
                # Minimum size to cover 1..30? 
                # k(k-1)/2 >= 30 => k ~ 8. So we can go down to 8.
                if len(neighbor) > 8:
                    idxs = [i for i, v in enumerate(neighbor) if v != 0]
                    if idxs:
                        idx = random.choice(idxs)
                        neighbor.pop(idx)
                        neighbor.sort()
                    else:
                        continue
            else:
                # Add an element (if size < 32)
                if len(neighbor) < 32:
                    max_val = max(neighbor)
                    new_val = random.randint(1, max_val)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                    else:
                        continue
            
            if len(neighbor) > 32 or 0 not in neighbor:
                continue
            
            # Check validity
            if not check_sidon(neighbor):
                continue
            
            n_diffs = get_diffs(neighbor)
            if not all(d in n_diffs for d in range(1, 31)):
                continue
            
            new_diam = max(neighbor)
            delta = new_diam - best_diam
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current = neighbor
                if new_diam < best_diam:
                    best = neighbor[:]
                    best_diam = new_diam
            
            T *= cooling

        return best

    # Main Logic
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Find a valid solution with a loose bound using MRV Backtracking
    # Try increasing diameter bounds until we find one
    # A lower bound for diameter: 
    # If we have k elements, max diff is D.
    # We need 30 distinct diffs.
    # Let's start with a reasonable guess. 
    # Random Sidon sets of size 10 usually have diameter ~ 100.
    # Let's try D from 40 upwards.
    
    found_initial = False
    for D in range(40, 101, 2):
        if time.time() > deadline - 60:
            break
            
        # Try a few random restarts of the MRV search? 
        # MRV is deterministic given the order. 
        # We can randomize the tie-breaking in MRV or the candidate order.
        # For now, let's just run it. If it fails, increase D.
        
        # To make MRV more robust, we can randomize the candidate iteration 
        # slightly or just rely on the "most covered" heuristic.
        
        res = mrv_backtracking(D, max_elements=32)
        if res:
            # Verify
            if check_sidon(res) and all(d in get_diffs(res) for d in range(1, 31)):
                found_initial = True
                best_set = res
                best_diam = max(res)
                break
    
    if not found_initial:
        # Fallback: Greedy construction
        # This is safer but might yield a larger diameter
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
        best_set = a
        best_diam = max(a)

    # Phase 2: Refine the diameter using Local Search
    if time.time() < deadline - 5:
        refined = refine_with_local_search(best_set, time_limit=deadline - 5)
        if check_sidon(refined) and all(d in get_diffs(refined) for d in range(1, 31)):
            if max(refined) < best_diam:
                best_set = refined
                best_diam = max(refined)

    # Phase 3: Try to squeeze further with targeted MRV on the current best
    # Try to find a solution with D = best_diam - 1, best_diam - 2, etc.
    if time.time() < deadline - 10:
        for D in range(best_diam - 1, max(30, best_diam - 10), -1):
            if time.time() > deadline - 10:
                break
            # Run MRV with this tighter bound
            # We might need to randomize the MRV choice to escape local traps
            found = False
            for _ in range(5): # 5 attempts
                if time.time() > deadline - 10:
                    break
                # Slight randomization in MRV: shuffle candidates with same score?
                # The current mrv_backtracking is deterministic. 
                # To allow different paths, we can modify it to accept a random seed 
                # or just rely on the fact that different D bounds change the search space.
                
                # Actually, mrv_backtracking as written is deterministic.
                # If it failed for D, it will fail again.
                # We need a randomized version of MRV or a different heuristic.
                
                # Let's implement a quick randomized greedy inside the loop for this phase
                # Or just skip if deterministic MRV already failed for this D.
                # Since mrv_backtracking is deterministic, calling it again with same D is useless.
                # We should only call it for NEW D values.
                
                res = mrv_backtracking(D, max_elements=32)
                if res:
                    if check_sidon(res) and all(d in get_diffs(res) for d in range(1, 31)):
                        if max(res) < best_diam:
                            best_set = res
                            best_diam = max(res)
                        found = True
                        break
            
            if found:
                continue # Try next smaller D
            else:
                # If we failed to find a solution for D, and D is close to best_diam,
                # it might be impossible, or MRV is too weak.
                # If D is significantly smaller, it's likely impossible.
                # But we should be cautious. Let's just stop if we fail a few in a row?
                # Or just break if we fail the first one? 
                # If we found a solution for best_diam, and fail for best_diam-1, 
                # it's likely best_diam is optimal or close.
                break

    return best_set

# EVOLVE-BLOCK-END
