"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill

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

    def count_violations(S):
        S_set = set(S)
        count = 0
        for a in S:
            for b in S:
                if a + b in S_set:
                    count += 1
        return count

    def greedy_max_sum_free(S, rng):
        """Greedy: repeatedly add the element that creates the fewest new violations."""
        S = list(S)
        rng.shuffle(S)
        
        # Start with a random odd subset as seed
        odd = [x for x in S if x % 2 != 0]
        if not odd:
            odd = S[:1]
        
        result = list(odd)
        result_set = set(result)
        
        # Compute initial violations
        # For each candidate not in result, count how many pairs (a,b) in result 
        # such that a+b == candidate, or candidate+a == b for some a,b in result
        # Actually simpler: just try adding each candidate and see if it creates violations
        
        remaining = [x for x in S if x not in result_set]
        
        # Sort remaining by "violation cost" - how many pairs in result would sum to it,
        # or how many pairs with it would sum to something in result
        def violation_cost(x, current_set):
            cost = 0
            for a in current_set:
                if a + x in current_set:
                    cost += 1
                if x + x in current_set:
                    cost += 1
            # Also check if x + a = b for a,b in current_set
            for a in current_set:
                b = x + a
                if b in current_set and b != x:
                    cost += 1
            return cost
        
        # Greedy: repeatedly pick the element with lowest violation cost that is 0
        # If none have cost 0, pick the one with lowest cost and add anyway (may need removal later)
        # Better: only add if cost is 0, otherwise skip
        
        added = True
        while added and len(remaining) > 0:
            added = False
            # Find all elements with cost 0
            zero_cost = []
            for x in remaining:
                c = violation_cost(x, result_set)
                if c == 0:
                    zero_cost.append((x, c))
            
            if zero_cost:
                # Pick one (random or first)
                idx = rng.randrange(len(zero_cost))
                x, _ = zero_cost[idx]
                result.append(x)
                result_set.add(x)
                remaining.remove(x)
                added = True
        
        return result

    def local_search_improve(S, rng, max_iters=2000):
        """Local search: start from greedy, try single-swap moves to increase size while staying sum-free."""
        current = greedy_max_sum_free(S, rng)
        current_set = set(current)
        best = list(current)
        best_size = len(best)
        
        for iteration in range(max_iters):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            improved = False
            # Try removing one element and adding another
            for i, rem in enumerate(current):
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                # Temporarily remove
                trial = current[:i] + current[i+1:]
                trial_set = set(trial)
                
                # Try adding each element not in trial_set
                candidates = [x for x in S if x not in trial_set]
                rng.shuffle(candidates)
                
                for add in candidates:
                    # Check if adding add keeps it sum-free
                    # Need to check: for all a,b in trial_set, a+b != add
                    # and add + a != b for all a,b in trial_set
                    valid = True
                    for a in trial_set:
                        if a + add in trial_set:
                            valid = False
                            break
                        if add + add in trial_set:
                            valid = False
                            break
                    if valid:
                        # Check add+a = b
                        for a in trial_set:
                            b = add + a
                            if b in trial_set and b != add:
                                valid = False
                                break
                    
                    if valid:
                        new_current = trial + [add]
                        if len(new_current) > best_size:
                            best = new_current
                            best_size = len(best)
                        current = new_current
                        current_set = set(current)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try single addition without removal
                candidates = [x for x in S if x not in current_set]
                rng.shuffle(candidates)
                for add in candidates:
                    valid = True
                    for a in current_set:
                        if a + add in current_set:
                            valid = False
                            break
                        if add + add in current_set:
                            valid = False
                            break
                    if valid:
                        for a in current_set:
                            b = add + a
                            if b in current_set and b != add:
                                valid = False
                                break
                    if valid:
                        current = current + [add]
                        current_set = set(current)
                        if len(current) > best_size:
                            best = current
                            best_size = len(best)
                        improved = True
                        break
            
            if not improved:
                break
        
        return best

    def solve_instance(S, rng):
        """Solve one instance: use multiple strategies and keep the best."""
        best = []
        best_size = 0
        
        # Strategy 1: Odd numbers
        odd = [x for x in S if x % 2 != 0]
        if is_sum_free(odd):
            best = odd
            best_size = len(odd)
        
        # Strategy 2: Greedy from random start
        for restart in range(20):
            candidate = greedy_max_sum_free(S, rng)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
            if time.time() - start_time > DEADLINE - 30:
                break
        
        # Strategy 3: Local search improvement
        if time.time() - start_time < DEADLINE - 20:
            candidate = local_search_improve(S, rng, max_iters=500)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
        
        # Strategy 4: Randomized greedy with different orderings
        for restart in range(30):
            if time.time() - start_time > DEADLINE - 15:
                break
            S_shuffled = list(S)
            rng.shuffle(S_shuffled)
            result = []
            result_set = set()
            for x in S_shuffled:
                # Check if x can be added
                valid = True
                for a in result_set:
                    if a + x in result_set:
                        valid = False
                        break
                    if x + x in result_set:
                        valid = False
                        break
                if valid:
                    for a in result_set:
                        b = x + a
                        if b in result_set and b != x:
                            valid = False
                            break
                if valid:
                    result.append(x)
                    result_set.add(x)
            if len(result) > best_size:
                best = result
                best_size = len(result)
        
        return best

    # Solve all three instances
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        S = instance(size, seed)
        rng = random.Random(seed + 12345)
        result = solve_instance(S, rng)
        results.append(result)
    
    # Final verification
    for i, result in enumerate(results):
        if not is_sum_free(result):
            # Fallback to odd numbers
            S = instance(instances[i][0], instances[i][1])
            results[i] = [x for x in S if x % 2 != 0]
    
    return results
# EVOLVE-BLOCK-END
