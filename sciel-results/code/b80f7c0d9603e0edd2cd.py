import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def solve_instance(S, time_limit):
        """
        Find maximum sum-free subset using multiple strategies.
        """
        n = len(S)
        S_set = set(S)
        
        # Precompute conflict relationships
        # x conflicts with y if x+y is in S, or x+x is in S (self-conflict)
        # For adding x to set T: x conflicts with T if:
        #   1. x + x in T
        #   2. x + a in T for some a in T  
        #   3. a + b = x for some a,b in T
        
        def try_add(x, T_set):
            """Check if x can be added to T_set while keeping it sum-free. O(k)."""
            if x + x in T_set:
                return False
            for a in T_set:
                if x + a in T_set:
                    return False
                if x - a in T_set:
                    return False
            return True
        
        def greedy_solve(order):
            """Greedy: add elements in given order if valid."""
            result = []
            result_set = set()
            for x in order:
                if try_add(x, result_set):
                    result.append(x)
                    result_set.add(x)
            return result
        
        def local_search(initial, max_iters=20000):
            """Aggressive local search with add/remove/swap moves."""
            current = list(initial)
            current_set = set(current)
            best = list(current)
            best_size = len(best)
            no_improve_count = 0
            
            for iteration in range(max_iters):
                if time.time() > time_limit:
                    break
                
                improved = False
                
                # Phase 1: Try to add any element
                candidates = [x for x in S if x not in current_set]
                random.shuffle(candidates)
                for add in candidates:
                    if try_add(add, current_set):
                        current.append(add)
                        current_set.add(add)
                        if len(current) > best_size:
                            best = list(current)
                            best_size = len(best)
                        improved = True
                        no_improve_count = 0
                        break
                
                if improved:
                    continue
                
                # Phase 2: Try swap - remove one, add one or more
                rem_candidates = list(current)
                random.shuffle(rem_candidates)
                
                for rem in rem_candidates:
                    if time.time() > time_limit:
                        break
                    
                    # Remove rem
                    trial_set = set(current)
                    trial_set.discard(rem)
                    trial_list = [x for x in current if x != rem]
                    
                    # Try to add as many as possible from remaining
                    add_candidates = [x for x in S if x not in trial_set]
                    random.shuffle(add_candidates)
                    
                    new_size = len(trial_list)
                    added = []
                    for add in add_candidates:
                        if try_add(add, trial_set):
                            trial_set.add(add)
                            added.append(add)
                            new_size += 1
                    
                    if new_size > best_size:
                        current = trial_list + added
                        current_set = trial_set
                        best = list(current)
                        best_size = new_size
                        improved = True
                        no_improve_count = 0
                        break
                    elif new_size == len(current) and added:
                        # Same size but different composition - accept with some probability
                        if random.random() < 0.3:
                            current = trial_list + added
                            current_set = trial_set
                            no_improve_count = 0
            
                if not improved:
                    no_improve_count += 1
                    if no_improve_count > 50:
                        break
            
            return best
        
        # Generate multiple diverse initial solutions
        best = []
        best_size = 0
        
        # Strategy 1: Positive elements sorted by magnitude (large positives tend to be sum-free together)
        pos = sorted([x for x in S if x > 0], key=lambda x: -x)
        candidate = greedy_solve(pos)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
        
        # Strategy 2: Negative elements sorted by magnitude
        neg = sorted([x for x in S if x < 0], key=lambda x: x)
        candidate = greedy_solve(neg)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
        
        # Strategy 3: Large absolute values
        large_abs = sorted(S, key=lambda x: -abs(x))
        candidate = greedy_solve(large_abs)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
        
        # Strategy 4: Random restarts
        for restart in range(30):
            if time.time() > time_limit:
                break
            order = list(S)
            random.shuffle(order)
            candidate = greedy_solve(order)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
        
        # Strategy 5: Local search from best solution
        if time.time() < time_limit - 3:
            candidate = local_search(best, max_iters=15000)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
        
        # Strategy 6: More local search restarts from diverse starts
        for restart in range(15):
            if time.time() > time_limit - 2:
                break
            
            # Create diverse starting points
            if restart % 3 == 0:
                # Start with positives
                order = sorted([x for x in S if x > 0], key=lambda x: -x)
                random.shuffle(order)
            elif restart % 3 == 1:
                # Start with negatives  
                order = sorted([x for x in S if x < 0], key=lambda x: x)
                random.shuffle(order)
            else:
                # Random
                order = list(S)
                random.shuffle(order)
            
            candidate = greedy_solve(order)
            candidate = local_search(candidate, max_iters=5000)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
        
        return best
    
    # Solve all three instances
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        S = instance(size, seed)
        time_remaining = DEADLINE - (time.time() - start_time)
        if time_remaining < 5:
            # Not enough time, use quick greedy with large abs
            order = sorted(S, key=lambda x: -abs(x))
            result = []
            result_set = set()
            for x in order:
                if x + x in result_set:
                    continue
                valid = True
                for a in result_set:
                    if x + a in result_set or x - a in result_set:
                        valid = False
                        break
                if valid:
                    result.append(x)
                    result_set.add(x)
            results.append(result)
        else:
            result = solve_instance(S, time.time() + time_remaining - 1)
            results.append(result)
    
    # Final verification
    for i, result in enumerate(results):
        if not is_sum_free(result):
            S = instance(instances[i][0], instances[i][1])
            results[i] = [x for x in S if x % 2 != 0]
    
    return results
