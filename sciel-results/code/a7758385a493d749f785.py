import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Helper: check if we can add element x to set a (with existing sums)
    def can_add(a, sums, x):
        for y in a:
            s = x + y
            if s in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def get_new_sums(a, x):
        return {x + y for y in a} | {2 * x}
    
    def compute_sums(a):
        s = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                s.add(x + y)
        return s
        
    def compute_conflicts(a, sums):
        """
        Returns a set of values x such that x cannot be added to a.
        """
        forbidden = set()
        for y in a:
            for s in sums:
                x = s - y
                if 1 <= x <= 284:
                    forbidden.add(x)
        for s in sums:
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= 284:
                    forbidden.add(x)
        return forbidden

    def fast_refill(a, sums, a_set, candidates, deadline):
        """
        Fast greedy refill. Picks valid candidates randomly to speed up search.
        """
        a = list(a)
        sums = set(sums)
        a_set = set(a_set)
        
        forbidden = compute_conflicts(a, sums)
        active_candidates = [c for c in candidates if c not in a_set and c not in forbidden]
        
        while active_candidates:
            if time.time() > deadline:
                break
            
            # Pick a random candidate to explore different paths quickly
            idx = random.randint(0, len(active_candidates) - 1)
            c = active_candidates[idx]
            
            a.append(c)
            a_set.add(c)
            new_sums = get_new_sums(a, c)
            sums |= new_sums
            
            forbidden = compute_conflicts(a, sums)
            
            # Update active candidates
            active_candidates = [x for x in active_candidates if x != c and x not in forbidden]
            
        return a, sums, a_set

    def greedy_expand_fast(initial_list, order, forbidden_cache=None):
        a = list(initial_list)
        sums = compute_sums(a)
        a_set = set(a)
        forbidden = compute_conflicts(a, sums)
        
        for x in order:
            if x in a_set:
                continue
            if x in forbidden:
                continue
            
            a.append(x)
            a_set.add(x)
            new_sums = get_new_sums(a, x)
            sums |= new_sums
            forbidden = compute_conflicts(a, sums)
            
        return a

    def strong_local_search(a, max_rounds=200, deadline=None):
        a = list(a)
        sums = compute_sums(a)
        a_set = set(a)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try to fill up using Fast Refill
            candidates = [x for x in range(1, 285) if x not in a_set]
            if candidates:
                prev_len = len(a)
                a, sums, a_set = fast_refill(a, sums, a_set, candidates, deadline)
                
                if len(a) > prev_len:
                    improved = True
                    continue
            
            # 2. If not improved, try removals
            non_fixed_indices = [i for i in range(len(a)) if a[i] not in fixed_set]
            
            if not non_fixed_indices:
                break
            
            # Try removals of size 1 to 4 (faster)
            best_candidate = None
            best_candidate_len = len(a)
            
            # Increase trials since refill is faster
            num_trials = 20
            
            for size in range(1, 5):
                if len(non_fixed_indices) < size:
                    continue
                
                for trial in range(num_trials):
                    if time.time() > deadline:
                        break
                    
                    # Pick random indices to remove
                    removed_indices = set(random.sample(non_fixed_indices, size))
                    removed_values = {a[i] for i in removed_indices}
                    
                    temp_a = [x for x in a if x not in removed_values]
                    temp_sums = compute_sums(temp_a)
                    temp_set = set(temp_a)
                    
                    # Fast Refill
                    refill_candidates = [x for x in range(1, 285) if x not in temp_set]
                    # No need to shuffle for fast_refill as it picks randomly internally
                    temp_a, temp_sums, temp_set = fast_refill(temp_a, temp_sums, temp_set, refill_candidates, deadline)
                    
                    if len(temp_a) > best_candidate_len:
                        best_candidate_len = len(temp_a)
                        best_candidate = (temp_a, temp_sums, temp_set)
                        
                        if best_candidate_len >= 20:
                            break
                
                if best_candidate_len >= 20:
                    break
                
            if best_candidate:
                a, sums, a_set = best_candidate
                improved = True
            else:
                break # No improvement found
            
        return a

    # Main search logic
    best = []
    
    # Strategy 1: Ascending order
    a1 = greedy_expand_fast(fixed_elements, range(12, 285))
    if len(a1) > len(best):
        best = a1
    
    # Strategy 2: Descending order
    a2 = greedy_expand_fast(fixed_elements, range(284, 11, -1))
    if len(a2) > len(best):
        best = a2

    # Strategy 3: Randomized greedy with local search
    random.seed(42)
    n_restarts = 500 # Increased restarts due to faster refill
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        a3 = greedy_expand_fast(fixed_elements, candidates)
        
        # Apply local search
        a3_improved = strong_local_search(a3, max_rounds=50, deadline=deadline)
        
        if len(a3_improved) > len(best):
            best = a3_improved
            if len(best) >= 20:
                break

    # Final polish
    best = strong_local_search(best, max_rounds=100, deadline=deadline)
    
    return best
