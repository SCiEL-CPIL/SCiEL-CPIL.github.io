"""Rewritten program for Sidon set with differences 1..30."""
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30.
    Minimize the largest element.
    
    Strategy: Use randomized local search with efficient Sidon checking,
    starting from multiple diverse initializations.
    """
    def get_diffs(arr):
        """Get all positive differences as a set."""
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(arr[j] - arr[i])
        return diffs
    
    def is_sidon_fast(arr):
        """Check if arr is a Sidon set using a set for O(n²) with early exit."""
        seen = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def has_all_diffs(arr, target_diffs):
        """Check if all target differences are present."""
        diff_set = get_diffs(arr)
        return all(d in diff_set for d in target_diffs)
    
    def objective(arr):
        """
        Objective: minimize max element, penalize missing differences and non-Sidon.
        Returns a score where lower is better.
        """
        if not is_sidon_fast(arr):
            # Count duplicate differences
            diff_list = []
            n = len(arr)
            for i in range(n):
                for j in range(i + 1, n):
                    diff_list.append(arr[j] - arr[i])
            dup_count = len(diff_list) - len(set(diff_list))
            return 10**9 + dup_count * 1000 + max(arr)
        
        diff_set = get_diffs(arr)
        missing = sum(1 for d in range(1, 31) if d not in diff_set)
        
        if missing > 0:
            return 10**6 + missing * 1000 + max(arr)
        
        return max(arr)
    
    def greedy_construct():
        """Greedy construction of a Sidon set with differences 1..30."""
        best_result = None
        best_obj = float('inf')
        
        for attempt in range(20):
            a = [0]
            missing_diffs = set(range(1, 31))
            
            # Try to build by adding elements that cover many missing diffs
            for step in range(15):  # We likely need at most ~12-15 elements
                if not missing_diffs:
                    break
                
                best_x = None
                best_score = float('inf')
                
                # Generate candidates: for each existing element and each missing diff
                candidates = set()
                for v in a:
                    for d in list(missing_diffs):
                        candidates.add(v + d)
                        if v - d >= 0:
                            candidates.add(v - d)
                
                # Add some random candidates
                max_val = max(a) if a else 0
                for _ in range(30):
                    candidates.add(random.randint(0, max_val + 30))
                
                candidates = [x for x in candidates if x >= 0 and x not in a]
                
                for x in candidates:
                    trial = sorted(a + [x])
                    if not is_sidon_fast(trial):
                        continue
                    
                    trial_diffs = get_diffs(trial)
                    new_missing = missing_diffs - trial_diffs
                    new_covered = len(missing_diffs) - len(new_missing)
                    
                    # Prefer candidates that cover more missing diffs, then smaller x
                    score = -new_covered * 10000 + x
                    
                    if score < best_score:
                        best_score = score
                        best_x = x
                
                if best_x is None:
                    # Try a wider range
                    for x in range(max_val + 1, max_val + 100):
                        if x in a:
                            continue
                        trial = sorted(a + [x])
                        if not is_sidon_fast(trial):
                            continue
                        trial_diffs = get_diffs(trial)
                        new_missing = missing_diffs - trial_diffs
                        new_covered = len(missing_diffs) - len(new_missing)
                        score = -new_covered * 10000 + x
                        if score < best_score:
                            best_score = score
                            best_x = x
                
                if best_x is None:
                    break
                
                a = sorted(a + [best_x])
                a_diffs = get_diffs(a)
                missing_diffs = missing_diffs - a_diffs
            
            if not missing_diffs and is_sidon_fast(a):
                obj = max(a)
                if obj < best_obj:
                    best_obj = obj
                    best_result = list(a)
        
        if best_result:
            return best_result
        
        # Fallback: simple sequential greedy
        a = [0]
        for d in range(1, 31):
            diff_set = get_diffs(a)
            if d in diff_set:
                continue
            found = False
            for x in range(0, max(a) + d + 20):
                if x in a:
                    continue
                trial = sorted(a + [x])
                if is_sidon_fast(trial):
                    a = trial
                    found = True
                    break
            if not found:
                x = 2 * max(a) + d + 1
                while not is_sidon_fast(sorted(a + [x])):
                    x += 1
                a = sorted(a + [x])
        
        return a
    
    def local_search(arr, max_iter=20000):
        """Local search to minimize max element while maintaining constraints."""
        best = sorted(arr)
        best_obj = objective(best)
        current = list(best)
        current_obj = best_obj
        
        for it in range(max_iter):
            # Pick a random element to move (not 0)
            idx = random.randint(1, len(current) - 1)
            old_val = current[idx]
            
            # Try moving to a new position, biased towards smaller values
            # Use a larger step size for more exploration
            step = random.randint(-80, 80)
            new_val = max(0, old_val + step)
            
            if new_val == old_val or new_val in current:
                continue
            
            trial = current[:]
            trial[idx] = new_val
            trial = sorted(trial)
            
            trial_obj = objective(trial)
            
            # Accept if better, or with probability for worse (simulated annealing)
            if trial_obj < current_obj:
                current = trial
                current_obj = trial_obj
                if current_obj < best_obj:
                    best = list(current)
                    best_obj = current_obj
            else:
                # Simulated annealing: accept worse with decreasing probability
                T = max(1, 200 * (1 - it / max_iter))
                if random.random() < (1 / (1 + (trial_obj - current_obj) / T)):
                    current = trial
                    current_obj = trial_obj
        
        return best
    
    # Try multiple random restarts
    best_overall = None
    best_obj_overall = float('inf')
    
    for restart in range(5):
        random.seed(restart * 1000 + 42)
        initial = greedy_construct()
        
        # Verify initial is valid
        if is_sidon_fast(initial) and has_all_diffs(initial, range(1, 31)):
            # Run local search to improve
            result = local_search(initial, max_iter=15000)
            
            # Final verification
            if is_sidon_fast(result) and has_all_diffs(result, range(1, 31)):
                obj = max(result)
                if obj < best_obj_overall:
                    best_obj_overall = obj
                    best_overall = sorted(result)
        
        # Also try the initial directly
        if is_sidon_fast(initial) and has_all_diffs(initial, range(1, 31)):
            obj = max(initial)
            if obj < best_obj_overall:
                best_obj_overall = obj
                best_overall = sorted(initial)
    
    if best_overall:
        return best_overall
    
    # Ultimate fallback
    return greedy_construct()
# EVOLVE-BLOCK-END
