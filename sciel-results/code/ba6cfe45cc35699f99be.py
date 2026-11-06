import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s buffer
    
    # Target interval
    T_MIN = 128
    T_MAX = 256
    N = 128
    
    # Helper to check coverage and multiplicity
    def evaluate(A):
        A_set = set(A)
        mult = [0] * (T_MAX + 1)
        covered = [False] * (T_MAX + 1)
        
        # Optimization: sort A
        A_sorted = sorted(A)
        for i, a in enumerate(A_sorted):
            # b >= a
            for b in A_sorted[i:]:
                s = a + b
                if T_MIN <= s <= T_MAX:
                    mult[s] += 1
                    covered[s] = True
                elif s > T_MAX:
                    break # Since A is sorted, further b will be larger
            
        # Check coverage
        for t in range(T_MIN, T_MAX + 1):
            if not covered[t]:
                return None, None, False # Invalid: not covered
        
        peak = max(mult[T_MIN:T_MAX+1])
        return A, peak, True

    # Helper to try to fix coverage by adding elements
    def try_fix_coverage(A):
        A_set = set(A)
        A_list = list(A)
        _, _, is_covered = evaluate(A_list)
        if is_covered:
            return A_list, True
        
        # Find missing targets
        mult = [0] * (T_MAX + 1)
        covered = [False] * (T_MAX + 1)
        A_sorted = sorted(A_list)
        for i, a in enumerate(A_sorted):
            for b in A_sorted[i:]:
                s = a + b
                if T_MIN <= s <= T_MAX:
                    covered[s] = True
                elif s > T_MAX:
                    break
        
        missing = [t for t in range(T_MIN, T_MAX + 1) if not covered[t]]
        
        # Try to add elements to cover missing targets
        # Greedy: for each missing t, find a candidate a to add
        # such that t-a is in A (or we add both? No, add one, other must be in A)
        # If t-a not in A, we can't add just a. We must add a such that t-a in A.
        # If no such a exists, we might need to add two? No, A is a set.
        # If we add a, and t-a is not in A, then a+t-a is not a sum of two elements in A yet.
        # So we must add a such that t-a is ALREADY in A.
        
        for t in missing:
            found = False
            # Candidates for a:
            # a must be in [0, N]
            # t-a must be in A_set
            # So a = t - b for some b in A_set
            # a must be in [0, N]
            candidates = []
            for b in A_set:
                a = t - b
                if 0 <= a <= N and a not in A_set:
                    candidates.append(a)
            
            if candidates:
                # Pick candidate that minimizes multiplicity increase
                best_a = None
                best_mult = float('inf')
                for a in candidates:
                    new_A = A_set.union([a])
                    # Quick check: does adding a create high multiplicity?
                    # Just count how many b in new_A such that a+b in [T_MIN, T_MAX]
                    # and how many existing sums a+b hit.
                    # Simplified: just pick one randomly or greedily
                    # For now, pick first
                    best_a = a
                    break # Greedy fast path
                
                if best_a is not None:
                    A_list.append(best_a)
                    A_set.add(best_a)
                    found = True
            
            if not found:
                # Cannot fix with single addition. 
                # This means for this t, for all b in A, t-b is not in [0,N] or is in A.
                # If t-b in A, it would be covered.
                # So t-b must be out of range or not in A.
                # If we can't fix, return invalid
                return A_list, False
        
        return A_list, True

    # Initialize A
    # Start with a random set that likely covers
    # Or start with seed and prune?
    # Let's start with a random set of size 30 and fix coverage.
    
    best_A = None
    best_peak = float('inf')
    
    # Try several random initializations
    for _ in range(50):
        if time.time() > deadline:
            break
            
        # Random subset
        A_init = random.sample(range(N + 1), 30)
        A_curr, is_valid = try_fix_coverage(A_init)
        
        if not is_valid:
            continue
            
        A_eval, peak, _ = evaluate(A_curr)
        if peak < best_peak:
            best_peak = peak
            best_A = A_curr
            
        if best_peak == 1:
            break
            
        # Local Search / Optimization
        # Try to remove elements or swap to reduce peak
        for step in range(500):
            if time.time() > deadline:
                break
                
            A_curr, peak, _ = evaluate(A_curr)
            if peak < best_peak:
                best_peak = peak
                best_A = A_curr
                if best_peak == 1:
                    break
            
            if peak == 1:
                break
                
            # Find t with max multiplicity
            mult = [0] * (T_MAX + 1)
            A_sorted = sorted(A_curr)
            for i, a in enumerate(A_sorted):
                for b in A_sorted[i:]:
                    s = a + b
                    if T_MIN <= s <= T_MAX:
                        mult[s] += 1
                    elif s > T_MAX:
                        break
                        
            max_mult = peak
            t_max = -1
            for t in range(T_MIN, T_MAX + 1):
                if mult[t] == max_mult:
                    t_max = t
                    break
            
            if t_max == -1:
                break
                
            # Try to remove an element involved in t_max
            pairs_summing_to_t_max = []
            A_set = set(A_curr)
            for a in A_curr:
                b = t_max - a
                if b in A_set and a <= b:
                    pairs_summing_to_t_max.append((a, b))
            
            improved = False
            for a, b in pairs_summing_to_t_max:
                # Try removing a
                if len(A_curr) > 1: # Keep at least 1? No, must cover.
                    A_temp = [x for x in A_curr if x != a]
                    A_temp_eval, peak_temp, is_covered = evaluate(A_temp)
                    if is_covered and peak_temp < peak:
                        A_curr = A_temp
                        improved = True
                        break
                        
                # Try removing b (if b != a)
                if b != a:
                    A_temp = [x for x in A_curr if x != b]
                    A_temp_eval, peak_temp, is_covered = evaluate(A_temp)
                    if is_covered and peak_temp < peak:
                        A_curr = A_temp
                        improved = True
                        break
            
            if not improved:
                # Try adding a new element to break collision?
                # Or swapping?
                # If we can't remove, maybe we can swap a with a_new
                # such that a_new doesn't contribute to t_max but helps elsewhere.
                # This is complex. Let's just do random restarts if stuck.
                break
                
    # If best_A is None, fallback to seed
    if best_A is None:
        return list(range(64, 129))
        
    return best_A

# Run construct to test? No, just define it.
