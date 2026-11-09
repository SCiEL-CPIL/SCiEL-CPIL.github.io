import time
import sys
import random
import numpy as np

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

    def calculate_score(arr):
        """
        Returns a score where lower is better.
        Components:
        1. Diameter (max element) - primary objective
        2. Penalty for missing differences (1..30)
        3. Penalty for non-Sidon property (should be 0 for valid sets)
        """
        if not arr:
            return float('inf')
        
        is_sidon, covered, _ = check_constraints(arr)
        
        if not is_sidon:
            # Heavy penalty for invalid Sidon sets
            return 1000000 + len(arr) * 1000 + max(arr)
            
        missing_count = 30 - len(covered)
        
        # We want to minimize max(arr).
        # But we must cover all 30 diffs.
        # If missing_count > 0, we prioritize covering them?
        # Actually, if we have a valid Sidon set with missing diffs, it's not a solution.
        # The objective is to find a VALID solution (missing_count == 0) with min max(arr).
        
        # Score = max(arr) + missing_count * BIG_PENALTY
        # If we find a valid set, missing_count is 0, so score = max(arr).
        # If we have invalid/missing, score is huge.
        
        if missing_count == 0:
            return max(arr)
        else:
            return 1000000 + missing_count * 10000 + max(arr)

    def _sa_construct():
        """
        Simulated Annealing to find a low-diameter Sidon set covering 1..30.
        """
        # Initialize with a random valid Sidon set?
        # Or start with a greedy one?
        # Let's start with a greedy construction to get a valid baseline, then SA to improve diameter.
        
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
                
                # Sort candidates by value to prefer smaller ones
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
                                pass
                        if not valid:
                            break
                    
                    if valid:
                        is_sidon, new_covered, new_used = check_constraints(trial)
                        if is_sidon:
                            a = trial
                            covered = new_covered
                            used_diffs = new_used
                            found = True
                            break
                
                if not found:
                    # Add two elements
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
                    # Force add (should be rare)
                    x = max(a) + d + 1
                    y = x + d
                    a = a + [x, y]
                    a.sort()
                    is_sidon, new_covered, new_used = check_constraints(a)
                    covered = new_covered
                    used_diffs = new_used
            
            return a

        # Get initial valid solution
        current_arr = greedy_construct()
        current_score = calculate_score(current_arr)
        best_arr = current_arr[:]
        best_score = current_score
        
        # SA Parameters
        T = 1000.0
        T_min = 1.0
        cooling = 0.999
        max_iters = 100000
        iter_count = 0
        
        while T > T_min and time.time() < deadline:
            # Generate neighbor
            # Move: Swap an element with a random new element in [0, 2*max]
            # Or: Replace an element with a random one.
            
            # To maintain Sidon property during SA is hard.
            # Strategy: 
            # 1. Pick a random index i in current_arr (except index 0 which is 0).
            # 2. Pick a random new value v in [0, upper_bound].
            # 3. If v is already in set, skip.
            # 4. Create new_arr = current_arr with arr[i] replaced by v, sorted.
            # 5. Calculate new_score.
            # 6. Accept if better, or with probability exp((current-new)/T).
            
            # Upper bound for random value:
            # Start with current max, maybe expand if stuck?
            # Let's use current max * 1.5 or best_score * 1.2
            
            upper_bound = max(100, int(best_score * 1.5))
            
            if len(current_arr) <= 1:
                continue
                
            idx = random.randint(1, len(current_arr) - 1) # Don't change 0
            old_val = current_arr[idx]
            
            # Try a few random values to increase chance of finding a valid move
            accepted = False
            for _ in range(5):
                new_val = random.randint(0, upper_bound)
                if new_val == old_val:
                    continue
                if new_val in current_arr:
                    continue
                
                new_arr = current_arr[:]
                new_arr[idx] = new_val
                new_arr.sort()
                
                # Quick check: if new_arr has duplicates, it's not a set.
                # But we checked 'in current_arr'.
                # However, replacing one element doesn't create duplicates with others unless new_val equals another element.
                # We checked 'in current_arr', so no duplicates.
                
                new_score = calculate_score(new_arr)
                
                # Acceptance criterion
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_arr = new_arr
                    current_score = new_score
                    accepted = True
                    break
            
            if not accepted:
                # Maybe try a different move? Or just cool down?
                pass
            
            if current_score < best_score:
                best_score = current_score
                best_arr = current_arr[:]
                
            T *= cooling
            iter_count += 1
            
            # Early stop if we found a very good solution?
            # Theoretical min for 30 diffs?
            # If best_score < 60, it's likely optimal or near optimal.
            if best_score < 50:
                break
                
        return best_arr

    # Run SA
    best_solution = _sa_construct()
    
    # Final verification
    is_sidon, covered, _ = check_constraints(best_solution)
    if not is_sidon or len(covered) < 30:
        # Fallback to greedy if SA failed to maintain validity (shouldn't happen with score penalty, but just in case)
        def greedy_construct():
            a = [0]
            covered = set()
            used_diffs = set()
            
            for d in range(1, 31):
                if d in covered:
                    continue
                
                found = False
                candidates = set()
                for a_i in a:
                    candidates.add(a_i + d)
                    candidates.add(a_i - d)
                
                for x in sorted(candidates):
                    if x in a or x < 0:
                        continue
                    trial = a + [x]
                    trial.sort()
                    new_diffs = []
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i+1, len(trial)):
                            diff = trial[j] - trial[i]
                            if diff in used_diffs:
                                valid = False
                                break
                            new_diffs.append(diff)
                        if not valid:
                            break
                    
                    if valid:
                        is_sidon, new_covered, new_used = check_constraints(trial)
                        if is_sidon:
                            a = trial
                            covered = new_covered
                            used_diffs = new_used
                            found = True
                            break
                
                if not found:
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
                    x = max(a) + d + 1
                    y = x + d
                    a = a + [x, y]
                    a.sort()
                    is_sidon, new_covered, new_used = check_constraints(a)
                    covered = new_covered
                    used_diffs = new_used
            
            return a
        best_solution = greedy_construct()
        
    return best_solution
# EVOLVE-BLOCK-END
