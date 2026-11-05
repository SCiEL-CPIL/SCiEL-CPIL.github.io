import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        if not subset:
            return True
        s_set = set(subset)
        n = len(subset)
        # Use set for O(1) lookup
        for i in range(n):
            a = subset[i]
            for j in range(i, n):
                c = a + subset[j]
                if c in s_set:
                    return False
        return True

    def can_add(C, C_set, x):
        # Check if x can be added to C
        if x in C_set:
            return False
        # 2x in C?
        if 2 * x in C_set:
            return False
        # x + c in C?
        for c in C:
            if x + c in C_set:
                return False
            if x - c in C_set:
                # This implies c + (x-c) = x, so x is a sum of two elements in C
                # Wait, if x-c is in C, then (x-c) + c = x.
                # So x is a sum of two elements in C.
                return False
        return True

    def best_k_subset(S):
        candidates = set([0])
        for x in S:
            candidates.add(abs(x))
        best_subset = []
        # We want to maximize size, so we want smallest K such that |x|>K is sum-free?
        # Not necessarily. Larger K means smaller set.
        # We want the LARGEST sum-free subset of this form.
        # So we iterate K from large to small?
        # If |x| > K is sum-free, then |x| > K' for K' > K is also sum-free (subset).
        # So we want the smallest K such that the set is sum-free.
        # Wait, if K is small, the set is large.
        # We want the largest set. So we want the smallest K.
        # But it might not be sum-free for small K.
        # So we iterate K from 0 upwards?
        # If K=0, set is all x != 0. Likely not sum-free.
        # If K is large, set is small. Likely sum-free.
        # We want the largest set, so we want the smallest K that works.
        
        # Let's just check all unique |x| values.
        for K in sorted(candidates):
            if time.time() - start_time > DEADLINE:
                break
            S_K = [x for x in S if abs(x) > K]
            if is_sum_free(S_K):
                if len(S_K) > len(best_subset):
                    best_subset = S_K
        return best_subset

    def greedy_max_abs(S):
        S_sorted = sorted(S, key=lambda x: abs(x), reverse=True)
        C = []
        C_set = set()
        for x in S_sorted:
            if can_add(C, C_set, x):
                C.append(x)
                C_set.add(x)
        return C

    def greedy_min_abs(S):
        S_sorted = sorted(S, key=lambda x: abs(x))
        C = []
        C_set = set()
        for x in S_sorted:
            if can_add(C, C_set, x):
                C.append(x)
                C_set.add(x)
        return C

    def greedy_value(S, reverse=True):
        S_sorted = sorted(S, key=lambda x: x, reverse=reverse)
        C = []
        C_set = set()
        for x in S_sorted:
            if can_add(C, C_set, x):
                C.append(x)
                C_set.add(x)
        return C

    def local_search(S, initial_subset, time_limit):
        C = list(initial_subset)
        C_set = set(C)
        remaining = [x for x in S if x not in C_set]
        remaining_set = set(remaining)
        
        if not is_sum_free(C):
            # Fallback to empty or simple fix
            return []

        best_size = len(C)
        no_improve_count = 0
        max_no_improve = 100 # Increased patience
        
        while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
            improved = False
            
            # 1. Try to insert any element
            for x in list(remaining):
                if time.time() - start_time > time_limit:
                    break
                if can_add(C, C_set, x):
                    C.append(x)
                    C_set.add(x)
                    remaining.remove(x)
                    remaining_set.discard(x)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue

            # 2. Try swap: Remove a, Add x
            # This keeps size same, but might enable further inserts.
            # Or, Remove a, Add x, Add y (size +1)
            
            found_move = False
            for i, a in enumerate(C):
                if time.time() - start_time > time_limit:
                    break
                if found_move:
                    break
                
                # Temporarily remove a
                C_set.discard(a)
                C_no_a = C[:i] + C[i+1:]
                
                added_any = False
                added_elements = []
                
                for x in list(remaining):
                    if time.time() - start_time > time_limit:
                        break
                    if can_add(C_no_a, C_set, x):
                        C_no_a.append(x)
                        C_set.add(x)
                        remaining.remove(x)
                        remaining_set.discard(x)
                        added_any = True
                        added_elements.append(x)
                        
                if added_any:
                    # If we added 1 or more, we improved or maintained size.
                    # If we added 1, size is same. We need to ensure we don't oscillate.
                    # If we added >1, size increased.
                    C = C_no_a
                    found_move = True
                    if len(added_elements) > 1:
                        no_improve_count = 0
                    else:
                        # Size same, but structure changed. Count as slight improvement?
                        # To avoid infinite loops, we can just accept it and hope for the best.
                        # Or, only accept if it allows a future insert?
                        # For now, accept any valid move that changes the set.
                        no_improve_count = 0 
                else:
                    # Restore a
                    C_set.add(a)
            
            if not found_move:
                no_improve_count += 1
        
        return C

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Generate structural candidates
        S_odd = [x for x in S if x % 2 != 0]
        
        # Modulo 3 candidates
        S_mod1 = [x for x in S if x % 3 == 1]
        S_mod2 = [x for x in S if x % 3 == 2]
        
        S_best_k = best_k_subset(S)
        
        M = max(S)
        m = min(S)
        S_upper = [x for x in S if x > M / 2.0]
        S_lower = [x for x in S if x < m / 2.0]
        
        S_greedy_max = greedy_max_abs(S)
        S_greedy_min = greedy_min_abs(S)
        S_greedy_val_desc = greedy_value(S, reverse=True)
        S_greedy_val_asc = greedy_value(S, reverse=False)
        
        S_pos = [x for x in S if x > 0]
        S_neg = [x for x in S if x < 0]
        
        S_sorted_val = sorted(S)
        half = len(S_sorted_val) // 2
        S_top_half = S_sorted_val[half:]
        S_bottom_half = S_sorted_val[:half]
        
        # Magnitude thresholds
        abs_vals = sorted(set(abs(x) for x in S))
        if abs_vals:
            S_mag_75 = [x for x in S if abs(x) > abs_vals[len(abs_vals)*3//4]]
            S_mag_50 = [x for x in S if abs(x) > abs_vals[len(abs_vals)//2]]
            S_mag_25 = [x for x in S if abs(x) > abs_vals[len(abs_vals)//4]]
        else:
            S_mag_75 = []
            S_mag_50 = []
            S_mag_25 = []
            
        candidates = [S_odd, S_mod1, S_mod2, S_best_k, S_upper, S_lower, 
                      S_greedy_max, S_greedy_min, S_greedy_val_desc, S_greedy_val_asc,
                      S_pos, S_neg, S_top_half, S_bottom_half, S_mag_75, S_mag_50, S_mag_25]
        
        # Filter to valid sum-free sets
        valid_candidates = []
        for cand in candidates:
            if is_sum_free(cand):
                valid_candidates.append(cand)
        
        if not valid_candidates:
            valid_candidates = [S_odd]
        
        best = max(valid_candidates, key=len)
        
        # Local search improvement
        if time.time() - start_time < DEADLINE - 5:
            improved = local_search(S, best, DEADLINE - 2)
            if len(improved) > len(best):
                best = improved
        
        # Final safety check
        if not is_sum_free(best):
            best = S_odd
        
        results.append(best)
    
    return results
