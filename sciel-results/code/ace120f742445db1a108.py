import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def check_constraints(arr):
        """
        Returns (is_sidon, covered_diffs_set, all_diffs_set)
        """
        n = len(arr)
        if n < 2:
            return True, set(), set()
        
        diffs = set()
        is_sidon = True
        
        # Check Sidon: all positive differences must be distinct
        # And collect covered diffs in 1..30
        # For efficiency, we can just compute all diffs and check uniqueness
        # n is at most 32, so n*(n-1)/2 <= 496 diffs. This is very fast.
        
        diff_list = []
        for i in range(n):
            ai = arr[i]
            for j in range(i+1, n):
                d = arr[j] - ai
                diff_list.append(d)
                if 1 <= d <= 30:
                    diffs.add(d)
        
        if len(diff_list) != len(set(diff_list)):
            is_sidon = False
            
        return is_sidon, diffs, set(diff_list)

    def get_missing_diffs(covered):
        return [d for d in range(1, 31) if d not in covered]

    def backtrack(arr, covered, used_diffs, depth, max_depth):
        """
        Recursive backtracking to complete the set.
        arr: current sorted list of elements
        covered: set of differences 1..30 already covered
        used_diffs: set of all differences generated so far (for Sidon check)
        depth: current recursion depth
        max_depth: max allowed size of the set
        """
        if time.time() > deadline:
            return None, None

        if not covered < set(range(1, 31)): # If all covered
             # Actually, we should check if we have all 1..30
             if len(covered) == 30:
                 return arr, covered
             # If we have all covered but haven't hit max_depth, we can stop?
             # No, we want to minimize max(arr). So if we have a valid solution, we return it.
             # But we are backtracking to find *a* solution. The caller will manage optimization.
             # For now, if we have all 30 diffs, this is a valid candidate.
             # However, we might be able to get a *smaller* max(arr) by rearranging?
             # Backtracking usually finds the first solution. To minimize max, we should
             # try to keep elements small.
             pass

        if depth >= max_depth:
            return None, None

        # If we have already exceeded the current best max, prune?
        # We don't pass 'best_max' here, so we rely on the search order.
        # We will generate candidates in increasing order of 'new_val' to find small solutions first.

        missing = get_missing_diffs(covered)
        if not missing:
            return arr, covered

        # Heuristic: Pick the most constrained missing difference?
        # Or just the smallest missing?
        # Smallest missing is usually best for keeping numbers small.
        target_d = min(missing)
        
        # We need to add one or more elements to realize target_d.
        # Adding one element x can realize target_d if:
        # 1. x - a_i = target_d for some a_i in arr  => x = a_i + target_d
        # 2. a_i - x = target_d for some a_i in arr  => x = a_i - target_d (but x >= 0 and x not in arr)
        # 3. Or we add two elements x, y such that y - x = target_d.
        
        # Strategy: Try adding a single element first.
        # Generate candidates for x based on existing elements.
        candidates = set()
        
        for a_i in arr:
            # Case 1: x > a_i, x - a_i = target_d
            x1 = a_i + target_d
            if x1 >= 0:
                candidates.add(x1)
            
            # Case 2: x < a_i, a_i - x = target_d
            x2 = a_i - target_d
            if x2 >= 0:
                candidates.add(x2)
        
        # Also, consider adding a "new" pair if single element addition fails or is not optimal?
        # If we can't find a single element, we must add two.
        # But let's try single elements first, sorted by value to keep max small.
        
        for x in sorted(candidates):
            if x in arr:
                continue
            
            # Check if adding x violates Sidon
            new_arr = arr + [x]
            new_arr.sort()
            
            # Quick Sidon check on new_arr
            # We can reuse the logic from check_constraints but just for Sidon
            is_sidon, new_covered, new_used_diffs = check_constraints(new_arr)
            
            if not is_sidon:
                continue
            
            # Recurse
            res_arr, res_covered = backtrack(new_arr, new_covered, new_used_diffs, depth + 1, max_depth)
            if res_arr is not None:
                return res_arr, res_covered
        
        # If no single element worked, try adding two elements x, y such that y - x = target_d.
        # This is more expensive. We can try to add a pair (x, x+target_d).
        # To keep numbers small, we should try small x.
        # But we must ensure x and x+target_d don't collide with existing diffs or each other.
        
        # Heuristic: Try x starting from 0, 1, 2... up to some limit?
        # Or derived from existing elements?
        # If we add x, it must not create a diff with existing elements that is already used.
        # And x+target_d must not create a diff with existing elements that is already used.
        # Also (x+target_d) - x = target_d is the new diff we want.
        
        # Let's try a limited range of x for the pair.
        # Since we want to minimize max, x should be small.
        # Max possible x? If max(arr) is M, we probably don't want x+target_d > M + some margin.
        # But we are constructing from scratch.
        
        # Let's try x in range(0, 50) + existing elements +/- small offsets?
        # Actually, if single element failed, it means for all a_i, a_i +/- target_d either:
        # 1. Was already in arr
        # 2. Created a duplicate difference with some other pair.
        
        # Adding a pair is a bigger jump.
        # Let's try to generate candidates for x in the pair (x, x+target_d).
        # x must be such that:
        # - x not in arr
        # - x+target_d not in arr
        # - For all a in arr: (x-a) and (a-x) not in used_diffs (if positive)
        # - For all a in arr: ((x+target_d)-a) and (a-(x+target_d)) not in used_diffs
        # - (x+target_d) - x = target_d not in used_diffs (it's missing, so it's not)
        
        # This check is expensive. Let's limit the search for x.
        # Try x in [0, 100] U [a_i - 50, a_i + 50] for all a_i?
        
        candidate_xs = set()
        for i in range(0, 100):
            candidate_xs.add(i)
        for a_i in arr:
            for offset in range(-20, 21):
                candidate_xs.add(a_i + offset)
        
        for x in sorted(candidate_xs):
            y = x + target_d
            if x in arr or y in arr:
                continue
            if y < 0:
                continue
            
            new_arr = arr + [x, y]
            new_arr.sort()
            
            is_sidon, new_covered, new_used_diffs = check_constraints(new_arr)
            if not is_sidon:
                continue
            
            res_arr, res_covered = backtrack(new_arr, new_covered, new_used_diffs, depth + 2, max_depth)
            if res_arr is not None:
                return res_arr, res_covered
        
        return None, None

    # Initial setup
    # We know 0 is in the set.
    # We want to cover 1..30.
    # Max size is 32.
    # Start with [0].
    
    # To improve performance, we can use a greedy initialization to get a valid solution,
    # then use backtracking to try to find a better (smaller max) one?
    # Or just use backtracking from the start with good heuristics.
    
    # Backtracking from [0] might be slow if the branching factor is high.
    # Let's try a hybrid: 
    # 1. Greedy to get a valid solution (fast).
    # 2. Use that to set an upper bound on max(arr).
    # 3. Backtrack with pruning: if max(current_arr) >= best_max, prune.
    
    def greedy_construct():
        a = [0]
        covered = set()
        used_diffs = set()
        
        for d in range(1, 31):
            if d in covered:
                continue
            
            found = False
            # Try to add one element
            candidates = set()
            for a_i in a:
                candidates.add(a_i + d)
                candidates.add(a_i - d)
            
            for x in sorted(candidates):
                if x in a or x < 0:
                    continue
                trial = a + [x]
                trial.sort()
                # Check Sidon
                new_diffs = []
                valid = True
                for i in range(len(trial)):
                    for j in range(i+1, len(trial)):
                        diff = trial[j] - trial[i]
                        if diff in used_diffs:
                            valid = False
                            break
                        new_diffs.append(diff)
                        if 1 <= diff <= 30:
                            pass # update covered later
                    if not valid:
                        break
                
                if valid:
                    # Update used_diffs and covered
                    # Re-calculate to be safe or update incrementally
                    # For simplicity in greedy, re-calc is fine for small n
                    is_sidon, new_covered, new_used = check_constraints(trial)
                    if is_sidon:
                        a = trial
                        covered = new_covered
                        used_diffs = new_used
                        found = True
                        break
            
            if not found:
                # Add two elements
                # Try small x
                for x in range(0, 100):
                    y = x + d
                    if x in a or y in a:
                        continue
                    trial = a + [x, y]
                    trial.sort()
                    is_sidon, new_covered, new_used = check_constraints(trial)
                    if is_sidon:
                        a = trial
                        covered = new_covered
                        used_diffs = new_used
                        found = True
                        break
            
            if not found:
                # Should not happen with d=1..30 and limit 32, but just in case
                # Force add
                x = max(a) + d + 1
                y = x + d
                a = a + [x, y]
                a.sort()
                is_sidon, new_covered, new_used = check_constraints(a)
                covered = new_covered
                used_diffs = new_used
        
        return a

    # 1. Get a valid solution with Greedy
    best_solution = greedy_construct()
    best_max = max(best_solution)
    
    # 2. Try to improve with Backtracking
    # We will try to find a solution with max < best_max
    # We can run backtracking multiple times with different random seeds or orderings?
    # The current backtrack is deterministic.
    # To make it stochastic, we can shuffle the order of candidates?
    # But we want small numbers, so sorted order is good.
    
    # Let's modify backtrack to accept a max_limit
    def backtrack_limited(arr, covered, used_diffs, depth, max_depth, max_limit):
        if time.time() > deadline:
            return None
        
        if max(arr) >= max_limit:
            return None
            
        if len(covered) == 30:
            return arr
        
        if depth >= max_depth:
            return None

        missing = get_missing_diffs(covered)
        if not missing:
            return arr

        target_d = min(missing)
        
        candidates = set()
        for a_i in arr:
            candidates.add(a_i + target_d)
            candidates.add(a_i - target_d)
        
        for x in sorted(candidates):
            if x in arr or x < 0 or x >= max_limit:
                continue
            
            new_arr = arr + [x]
            new_arr.sort()
            
            is_sidon, new_covered, new_used_diffs = check_constraints(new_arr)
            if not is_sidon:
                continue
            
            res = backtrack_limited(new_arr, new_covered, new_used_diffs, depth + 1, max_depth, max_limit)
            if res is not None:
                return res
        
        # Try pair
        candidate_xs = set()
        for i in range(0, max_limit):
            candidate_xs.add(i)
        for a_i in arr:
            for offset in range(-20, 21):
                v = a_i + offset
                if 0 <= v < max_limit:
                    candidate_xs.add(v)
        
        for x in sorted(candidate_xs):
            y = x + target_d
            if y >= max_limit:
                break
            if x in arr or y in arr:
                continue
            
            new_arr = arr + [x, y]
            new_arr.sort()
            
            is_sidon, new_covered, new_used_diffs = check_constraints(new_arr)
            if not is_sidon:
                continue
            
            res = backtrack_limited(new_arr, new_covered, new_used_diffs, depth + 2, max_depth, max_limit)
            if res is not None:
                return res
        
        return None

    # Try to find a solution with max < best_max
    # We can try decreasing max_limit
    # Or just run backtrack_limited with max_limit = best_max - 1
    
    if time.time() < deadline:
        # Try to improve
        # We can try several times with slightly different randomizations?
        # The current backtrack is deterministic.
        # Let's just try to find ANY solution with max < best_max.
        
        # To increase chance, we can randomize the order of 'target_d' selection?
        # No, picking smallest missing is usually best for compactness.
        
        # Let's try to run backtrack_limited with max_limit = best_max - 1
        # If it fails, try best_max - 2, etc? No, that's too slow.
        
        # Instead, let's just run the backtracking with the limit.
        # If it finds a solution, update best.
        
        # To make it more robust, we can try multiple random starting points?
        # No, we must include 0.
        
        # Let's just run the deterministic backtrack with the limit.
        # If it's too slow, the deadline guard will stop it.
        
        # Optimization: If best_max is already small (e.g., < 100), backtracking might be fast.
        # If best_max is large, it might be slow.
        
        # Let's try a few random restarts of the backtracking with different candidate orderings?
        # Actually, the candidate ordering (sorted) is fixed.
        
        # Let's just run it once.
        res = backtrack_limited([0], set(), set(), 0, 32, best_max)
        if res is not None:
            new_max = max(res)
            if new_max < best_max:
                best_solution = res
                best_max = new_max
    
    # Final check
    is_sidon, covered, _ = check_constraints(best_solution)
    if not is_sidon or len(covered) < 30:
        # Fallback to greedy if something went wrong (shouldn't happen)
        best_solution = greedy_construct()
        
    return best_solution
# EVOLVE-BLOCK-END
