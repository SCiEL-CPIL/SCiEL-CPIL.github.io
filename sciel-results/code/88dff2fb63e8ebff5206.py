import time
import random

# EVOLVE-BLOCK-START
P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Precompute prime factorizations for all numbers
    def prime_factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    factorizations = {}
    for i in range(2, n + 1):
        factorizations[i] = prime_factorize(i)
    
    def divides_product(a, b, c):
        """Check if a divides b*c using factorizations"""
        for p, exp_a in factorizations[a].items():
            exp_bc = factorizations[b].get(p, 0) + factorizations[c].get(p, 0)
            if exp_bc < exp_a:
                return False
        return True
        
    def can_add(S, new_elem):
        """
        Check if adding new_elem to set S keeps it valid.
        S is a list of ints. new_elem is not in S.
        """
        m = len(S)
        # Check condition 1: new_elem | a*b
        for i in range(m):
            a = S[i]
            for j in range(i + 1, m):
                b = S[j]
                if divides_product(new_elem, a, b):
                    return False
        
        # Check condition 2 & 3: a | new_elem*b or b | new_elem*a
        for i in range(m):
            a = S[i]
            for j in range(m):
                if i == j:
                    continue
                b = S[j]
                # Check a | new_elem * b
                if divides_product(a, new_elem, b):
                    return False
        return True

    def is_valid_set(S):
        """Check if set S is valid: no three distinct a,b,c with a|b*c"""
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if j == i:
                    continue
                b = S_list[j]
                for k in range(m):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if divides_product(a, b, c):
                        return False
        return True

    best_set = set()
    
    # Strategy: Multiple Randomized Greedy with different ordering strategies
    # and Local Search
    
    # Pre-defined sorting keys for diversity
    def key_few_factors(x):
        return (len(factorizations[x]), random.random())
        
    def key_large(x):
        return (-x, random.random())
        
    def key_large_primes(x):
        # Prefer numbers with large largest prime factor
        max_p = max(factorizations[x].keys()) if factorizations[x] else 0
        return (-max_p, random.random())
        
    def key_random(x):
        return random.random()
        
    strategies = [key_few_factors, key_large, key_large_primes, key_random]
    
    # Determine number of restarts based on time
    # We'll run restarts until time runs out, but at least a few
    min_restarts = 10
    max_restarts = 50
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        if r < min_restarts and time.time() > deadline - 5:
             break # Ensure we finish at least min_restarts if possible, else stop
            
        current_set = []
        current_set_set = set()
        
        # Select strategy
        strategy = strategies[r % len(strategies)]
        sorted_candidates = sorted(candidates, key=strategy)
        
        # Greedy Construction
        for a in sorted_candidates:
            if time.time() > deadline - 10:
                break
            if can_add(current_set, a):
                current_set.append(a)
                current_set_set.add(a)
                
        # Local Improvement
        improved = True
        # Limit local search steps to ensure time safety
        max_local_steps = 50
        steps = 0
        
        while improved and steps < max_local_steps and time.time() < deadline - 5:
            improved = False
            steps += 1
            
            # 1. Try Add
            # Iterate through candidates not in set
            # To speed up, we can shuffle or sort by heuristic
            # Here we just iterate all, but break if time is tight
            added_any = False
            # Create a list of candidates not in current_set
            # This is O(N), acceptable
            not_in_set = [x for x in candidates if x not in current_set_set]
            
            # Sort by some heuristic? Random is fine for diversity
            random.shuffle(not_in_set)
            
            for v in not_in_set:
                if time.time() > deadline - 5:
                    break
                if can_add(current_set, v):
                    current_set.append(v)
                    current_set_set.add(v)
                    added_any = True
                    # If we added one, others might be invalid, so restart add phase?
                    # For simplicity, we just add as many as possible in this pass.
                    # But can_add depends on current_set.
                    # So we should re-evaluate?
                    # If we add v, we don't re-check previous additions in this loop.
                    # This is standard.
                    pass
            
            if added_any:
                improved = True
                continue # Restart local search loop
            
            # 2. Try 1-for-2 Swap
            # Remove u, add v1, v2
            # Try a limited number of u's to save time
            # Pick random u's from current_set
            if current_set:
                # Shuffle current set to try different removals
                removal_order = current_set.copy()
                random.shuffle(removal_order)
                
                # Try first 10 removals
                swap_found = False
                for u in removal_order[:10]:
                    if time.time() > deadline - 5:
                        break
                    trial_set = [x for x in current_set if x != u]
                    trial_set_set = current_set_set - {u}
                    
                    added = []
                    # Try to add 2 elements
                    # Use the same 'not_in_set' logic but for trial_set
                    # Re-calculate not_in_set for trial_set
                    not_in_trial = [x for x in candidates if x not in trial_set_set]
                    random.shuffle(not_in_trial)
                    
                    for v in not_in_trial:
                        if len(added) >= 2:
                            break
                        if can_add(trial_set, v):
                            trial_set.append(v)
                            trial_set_set.add(v)
                            added.append(v)
                    
                    if len(added) == 2:
                        current_set = trial_set
                        current_set_set = trial_set_set
                        improved = True
                        swap_found = True
                        break
                
                if swap_found:
                    continue
            
            # 3. Try 1-for-1 Swap (Perturbation)
            # Only if we are stuck
            if time.time() < deadline - 10 and current_set:
                # Do a few random 1-for-1 swaps
                # This doesn't increase size, but might escape local optima
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    u = random.choice(current_set)
                    not_in_set = [x for x in candidates if x not in current_set_set]
                    if not not_in_set:
                        break
                    v = random.choice(not_in_set)
                    trial_set = [x for x in current_set if x != u]
                    if can_add(trial_set, v):
                        trial_set.append(v)
                        current_set = trial_set
                        current_set_set = set(trial_set)
                        # We changed the set, so we should re-evaluate
                        # Set improved to True to continue the while loop?
                        # But size didn't change.
                        # Let's just break out of the 1-for-1 loop and let the while loop
                        # re-evaluate adds and swaps.
                        # To do that, we need to signal that we changed state.
                        # We can set improved = True, but that implies size increase.
                        # Instead, we just break out of the inner loop and let the while
                        # loop condition check time and 'improved' (which is False).
                        # But we want to continue searching!
                        # So we should set a flag 'perturbed' and break.
                        # Then the while loop will restart, and 'improved' is False,
                        # so it will terminate.
                        # To avoid termination, we can set improved = True.
                        # This is a hack, but it allows more exploration.
                        improved = True
                        break
        
        # End Local Search
        
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            
        # If we have time, we could try more restarts
        # The outer loop handles this.

    # Final Verification
    if not is_valid_set(best_set):
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    # Ensure all elements are Python ints
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
