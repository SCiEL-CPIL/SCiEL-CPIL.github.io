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

    def smart_refill(a, sums, a_set, candidates, deadline):
        """
        Refills the set using a 'minimum impact' greedy heuristic.
        At each step, among valid candidates, pick the one that forbids the fewest other candidates.
        """
        a = list(a)
        sums = set(sums)
        a_set = set(a_set)
        
        # Precompute which candidates are currently valid
        forbidden = compute_conflicts(a, sums)
        
        # Filter candidates to those not in a and not forbidden
        # We will maintain a list of active candidates
        active_candidates = [c for c in candidates if c not in a_set and c not in forbidden]
        
        while active_candidates:
            if time.time() > deadline:
                break
            
            best_cand = None
            min_impact = float('inf')
            best_new_forbidden = None
            best_new_sums = None
            best_new_a = None
            
            # Evaluate each active candidate
            for c in active_candidates:
                # c is already known to be valid (not in forbidden)
                # Calculate new sums
                new_sums_candidate = get_new_sums(a, c)
                # Check if new_sums_candidate introduces any conflict with existing sums?
                # No, 'c not in forbidden' ensures x+y not in sums and 2x not in sums.
                # So it is a valid addition.
                
                # Calculate impact: how many other active candidates become forbidden?
                # New forbidden set would be:
                # Old forbidden + {c} + {y | y+c in sums OR y+z in new_sums_candidate OR 2y in new_sums_candidate ...}
                # Actually, simpler:
                # A candidate c2 is forbidden if:
                # 1. c2 + y = s for some y in a_new, s in sums_new
                # 2. 2*c2 = s for some s in sums_new
                
                # We can approximate impact by checking which active_candidates c2
                # would be invalid against the NEW state.
                
                # To save time, we only check a subset of active_candidates if it's large?
                # No, 280 is small.
                
                impact = 0
                new_a_set_temp = a_set | {c}
                new_sums_temp = sums | new_sums_candidate
                
                # We don't need to compute full new_forbidden set for every candidate,
                # just count how many active_candidates are killed.
                
                # Optimization: 
                # c2 is killed if:
                # - c2 + y in sums_new for some y in a_new
                # - 2*c2 in sums_new
                
                # Let's just iterate active_candidates (excluding c)
                for c2 in active_candidates:
                    if c2 == c:
                        continue
                    # Check if c2 is valid in new state
                    valid = True
                    # Check sums with a
                    for y in a:
                        if c2 + y in new_sums_temp: # Note: sums in new_sums_temp includes old sums + new sums
                            valid = False
                            break
                    if valid:
                        if c2 + c in new_sums_temp: # y=c
                            valid = False
                        elif 2 * c2 in new_sums_temp:
                            valid = False
                    
                    if not valid:
                        impact += 1
                        
                if impact < min_impact:
                    min_impact = impact
                    best_cand = c
                    best_new_forbidden = None # We'll recompute or update
                    best_new_sums = new_sums_temp
                    best_new_a = a + [c]
                    
            if best_cand is None:
                break
                
            # Commit the best candidate
            a = best_new_a
            a_set.add(best_cand)
            sums = best_new_sums
            
            # Update forbidden set incrementally or recompute?
            # Recompute is safe and fast enough for N=284
            forbidden = compute_conflicts(a, sums)
            
            # Update active_candidates: remove best_cand and any that became forbidden
            active_candidates = [c for c in active_candidates if c != best_cand and c not in forbidden]
            
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

    def strong_local_search(a, max_rounds=50, deadline=None):
        a = list(a)
        sums = compute_sums(a)
        a_set = set(a)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try to fill up using Smart Refill
            candidates = [x for x in range(1, 285) if x not in a_set]
            if candidates:
                prev_len = len(a)
                a, sums, a_set = smart_refill(a, sums, a_set, candidates, deadline)
                
                if len(a) > prev_len:
                    improved = True
                    # If we improved, we might be able to add more?
                    # smart_refill runs until no more can be added.
                    # So if it returned, it's stuck.
                    # But maybe a different order of removal would help?
                    # We just accept this improvement and move to next round?
                    # Or try removals immediately?
                    # Let's try removals in the same round if we improved?
                    # No, let's just go to next round. The next round will try removals.
                    continue
            
            # 2. If not improved, try removals
            non_fixed_indices = [i for i in range(len(a)) if a[i] not in fixed_set]
            
            if not non_fixed_indices:
                break
            
            # Try removals of size 1 to 6
            best_candidate = None
            best_candidate_len = len(a)
            
            # Number of trials per size
            num_trials = 10
            
            for size in range(1, 7):
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
                    
                    # Smart Refill
                    refill_candidates = [x for x in range(1, 285) if x not in temp_set]
                    # Randomize order to explore different paths
                    random.shuffle(refill_candidates)
                    
                    temp_a, temp_sums, temp_set = smart_refill(temp_a, temp_sums, temp_set, refill_candidates, deadline)
                    
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
    n_restarts = 100
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        a3 = greedy_expand_fast(fixed_elements, candidates)
        
        # Apply local search
        a3_improved = strong_local_search(a3, max_rounds=20, deadline=deadline)
        
        if len(a3_improved) > len(best):
            best = a3_improved
            if len(best) >= 20:
                break

    # Final polish
    best = strong_local_search(best, max_rounds=50, deadline=deadline)
    
    return best
