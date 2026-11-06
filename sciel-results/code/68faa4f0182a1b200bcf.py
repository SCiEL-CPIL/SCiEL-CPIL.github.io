"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        if len(S) < 3:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def can_add(S_set, c):
        """Check if c can be added to S_set while maintaining sum-free property."""
        if c in S_set:
            return False
        # Check: 2*c in S (c + c = 2c)
        if 2 * c in S_set:
            return False
        # Check: for each a in S, a + c not in S and c - a not in S
        for a in S_set:
            if a + c in S_set:
                return False
            if c - a in S_set:
                return False
        return True
    
    def greedy_build(S_init, candidates, time_limit=1.0):
        """
        Greedily add elements from candidates to S_init while maintaining sum-free property.
        Returns (new_list, new_set).
        """
        S_set = set(S_init) if S_init is not None else set()
        result = list(S_init) if S_init is not None else []
        s_list = list(S_set)
        
        # Precompute conflicts: for each candidate, track which existing elements conflict
        # A candidate c conflicts with S if:
        # 1. 2*c in S
        # 2. exists a in S such that a+c in S or c-a in S
        
        # Build conflict sets for efficiency
        # For each element a in S, the "forbidden" candidates are:
        # - a (itself, already in S)
        # - -a (since a + (-a) = 0, but 0 not in range, so this is fine)
        # - Actually, let's think differently:
        # c is valid iff:
        #   2c not in S
        #   for all a in S: (a+c) not in S AND (c-a) not in S
        
        # Let's precompute: for each a in S, what values of c would be invalid?
        # Invalid if a+c in S, i.e., c in {s - a for s in S}
        # Invalid if c-a in S, i.e., c in {s + a for s in S}
        # Invalid if 2c in S, i.e., c in {s/2 for s in S} (only if s even)
        
        # This is getting complex. Let's just do direct checking but optimize the inner loop.
        
        remaining_time = time_limit
        for c in candidates:
            if time.time() - start_time > deadline:
                break
            if c in S_set:
                continue
            
            # Quick check: 2*c in S_set
            if 2 * c in S_set:
                continue
            
            # Check pairwise conflicts
            valid = True
            for a in s_list:
                if a + c in S_set or c - a in S_set:
                    valid = False
                    break
            
            if valid:
                result.append(c)
                S_set.add(c)
                s_list.append(c)
        
        return result, S_set
    
    def local_search(S_list, values, time_limit=5.0):
        """
        Improve S using local search:
        1. Try greedy additions
        2. Try removing one element and adding multiple others
        3. Try swap moves
        """
        S_set = set(S_list)
        ls_start = time.time()
        
        # Phase 1: Greedy additions
        candidates = [x for x in values if x not in S_set]
        S_list, S_set = greedy_build(S_list, candidates, time_limit=0.5)
        
        # Phase 2: Local search - remove one, try to add multiple
        improved = True
        while improved and time.time() - ls_start < time_limit:
            improved = False
            S_list_copy = list(S_list)
            S_set_copy = set(S_set)
            
            # Try removing each element and see if we can add more
            for x in S_list_copy:
                if time.time() - ls_start > time_limit:
                    break
                
                # Remove x
                S_test = S_set_copy - {x}
                S_list_test = [y for y in S_list_copy if y != x]
                
                # Try to add all excluded elements greedily
                excluded = [y for y in values if y not in S_test and y != x]
                S_list_new, S_set_new = greedy_build(S_list_test, excluded, time_limit=0.3)
                
                if len(S_set_new) > len(S_set):
                    S_list = S_list_new
                    S_set = S_set_new
                    improved = True
                    break
            
            # Phase 2b: Swap moves
            if not improved and time.time() - ls_start < time_limit:
                excluded = [y for y in values if y not in S_set]
                for x in S_list:
                    if time.time() - ls_start > time_limit:
                        break
                    
                    S_test = S_set - {x}
                    S_list_test = [y for y in S_list if y != x]
                    
                    for y in excluded:
                        if y in S_test:
                            continue
                        if can_add(S_test, y):
                            # Swap x for y
                            S_list_new = S_list_test + [y]
                            S_set_new = S_test | {y}
                            # Try to add more
                            remaining = [z for z in values if z not in S_set_new and z != x]
                            S_list_new, S_set_new = greedy_build(S_list_new, remaining, time_limit=0.2)
                            if len(S_set_new) > len(S_set):
                                S_list = S_list_new
                                S_set = S_set_new
                                improved = True
                                break
                    if improved:
                        break
        
        return S_list
    
    def solve_one(values, time_budget=30):
        """Find a large sum-free subset of values."""
        if time.time() > deadline:
            return []
        
        n = len(values)
        if n == 0:
            return []
        
        # Strategy 1: Start with odd numbers (always sum-free)
        odd_subset = [x for x in values if x % 2 != 0]
        
        # Strategy 2: Start with numbers in a large range
        # If we take all numbers > n/2 in absolute value, they're likely sum-free
        # Actually, let's try: numbers with |x| > n/2
        large_subset = [x for x in values if abs(x) > n // 2]
        
        # Strategy 3: Random initial set
        rng = random.Random(123)
        random_subset = rng.sample(values, min(len(values), max(10, len(values) // 3)))
        
        # Try multiple starting points and greedy builds
        best = []
        best_set = set()
        
        initial_sets = [odd_subset, large_subset, random_subset, []]
        
        for init in initial_sets:
            if time.time() > deadline:
                break
            
            # Try greedy from this initial set
            candidates = [x for x in values if x not in set(init)]
            trial_list, trial_set = greedy_build(init, candidates, time_limit=1.0)
            
            if len(trial_list) > len(best):
                best = trial_list
                best_set = trial_set
        
        # Local search on best
        remaining_time = deadline - time.time() - 2
        if remaining_time > 3:
            ls_time = min(remaining_time - 2, time_budget * 0.4)
            improved = local_search(best, values, time_limit=ls_time)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)
        
        # Try random restarts
        rng = random.Random(42)
        remaining_time = deadline - time.time() - 3
        num_restarts = max(1, int(remaining_time / 4))
        for i in range(num_restarts):
            if time.time() > deadline - 3:
                break
            
            shuffled = list(values)
            rng.shuffle(shuffled)
            
            # Random initial subset
            init_size = max(5, len(values) // 4)
            init_subset = shuffled[:init_size]
            
            trial_list, trial_set = greedy_build(init_subset, shuffled[init_size:], time_limit=0.8)
            
            if len(trial_list) > len(best):
                # Local search on this new best
                ls_time = min(2.0, deadline - time.time() - 3)
                if ls_time > 1:
                    improved = local_search(trial_list, values, time_limit=ls_time)
                    if len(improved) > len(best):
                        best = improved
                        best_set = set(best)
                elif len(trial_list) > len(best):
                    best = trial_list
                    best_set = trial_set
        
        return best
    
    # Process instances
    instances = [(60, 801), (120, 802), (240, 803)]
    total_size = sum(s for s, _ in instances)
    
    results = []
    for size, seed in instances:
        if time.time() > deadline:
            results.append([])
            continue
        
        values = instance(size, seed)
        
        # Allocate time proportional to size
        time_budget = max(15, (size / total_size) * 130)
        subset = solve_one(values, time_budget=time_budget)
        
        # Final verification and fallback
        if not is_sum_free(subset):
            # Fallback to odd numbers
            subset = [x for x in values if x % 2 != 0]
        
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
