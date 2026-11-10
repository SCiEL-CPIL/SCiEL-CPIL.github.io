# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 80  # Leave 10s margin for safety
    best_set = []
    
    # Helper to compute dp bitsets for a list of numbers
    def compute_dp(nums):
        n = len(nums)
        if n == 0:
            return [1] # dp[0] = 1 (bit 0)
        max_sum = sum(nums)
        # dp[k] is an integer where bit s is 1 if there's a subset of size k summing to s
        dp = [0] * (n + 1)
        dp[0] = 1  # bit 0 is 1
        for x in nums:
            # Update in reverse order of k to avoid using x multiple times
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        return dp

    def is_valid_set(S):
        """
        Full validation of a set S.
        Returns True if no element in S is the mean of any subset of at least two OTHER elements in S.
        """
        if len(S) < 3:
            return True
        
        n = len(S)
        dp = compute_dp(S)
        
        for i, a in enumerate(S):
            # Check if a is the mean of any subset of S \ {a} of size k >= 2
            # Sum(subset) = k * a
            # We need to check subsets of S \ {a}.
            S_excl_a = [x for x in S if x != a]
            if len(S_excl_a) < 2:
                continue
            dp_excl = compute_dp(S_excl_a)
            m = len(S_excl_a)
            for k in range(2, m + 1):
                target_sum = k * a
                if target_sum > sum(S_excl_a):
                    break
                if (dp_excl[k] >> target_sum) & 1:
                    return False
        return True

    def count_violations(S):
        """
        Count the number of violated mean constraints in set S.
        A violation occurs when an element a in S is the exact arithmetic mean
        of some subset T of S\{a} with |T| >= 2.
        """
        if len(S) < 3:
            return 0
        
        n = len(S)
        total_violations = 0
        
        for i, a in enumerate(S):
            S_excl_a = [x for x in S if x != a]
            if len(S_excl_a) < 2:
                continue
            dp_excl = compute_dp(S_excl_a)
            m = len(S_excl_a)
            for k in range(2, m + 1):
                target_sum = k * a
                if target_sum > sum(S_excl_a):
                    break
                if (dp_excl[k] >> target_sum) & 1:
                    total_violations += 1
                    break  # Only count each element once if it has any violation
        
        return total_violations

    def is_valid_addition(S, x):
        """
        Check if x can be added to S without violating constraints.
        S is a list of ints. x is not in S.
        Returns True if valid.
        """
        if x in S:
            return False
        
        n = len(S)
        if n == 0:
            return True
        
        # 1. Check if x is the mean of any subset of S of size k >= 2
        # sum(T) = k * x
        dp_S = compute_dp(S)
        for k in range(2, n + 1):
            target_sum = k * x
            if target_sum > sum(S):
                break
            if (dp_S[k] >> target_sum) & 1:
                return False
                
        # 2. Check if any a in S is the mean of a subset of S U {x} containing x
        # Subset T U {x}, T subset of S \ {a}
        # sum(T) + x = (|T| + 1) * a
        # sum(T) = (|T| + 1) * a - x
        for a in S:
            S_excl_a = [y for y in S if y != a]
            if not S_excl_a:
                continue
            dp_excl = compute_dp(S_excl_a)
            m = len(S_excl_a)
            for j in range(1, m + 1):
                target_sum = (j + 1) * a - x
                if target_sum < 0:
                    continue
                if target_sum > sum(S_excl_a):
                    break 
                if (dp_excl[j] >> target_sum) & 1:
                    return False
                    
        return True

    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True

    def greedy_construct(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= 48:
                    break
        return selected

    def simulated_annealing(initial_set, max_iter=5000):
        """
        Simulated Annealing to maximize set size while maintaining validity.
        State: a set of integers from [1, 512]
        Moves: add a random element, remove a random element, or swap
        Objective: maximize |S| - lambda * violations(S)
        """
        current_set = list(initial_set)
        current_violations = count_violations(current_set)
        best_set = list(current_set)
        best_violations = current_violations
        best_size = len(current_set)
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
        T = T_start
        
        all_numbers = list(range(1, 513))
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Generate a random move
            move_type = random.random()
            new_set = list(current_set)
            
            if move_type < 0.4:
                # Add a random element
                if len(current_set) < 48:
                    available = [x for x in all_numbers if x not in current_set]
                    if available:
                        x = random.choice(available)
                        new_set.append(x)
            elif move_type < 0.7:
                # Remove a random element
                if len(current_set) > 1:
                    idx = random.randint(0, len(current_set) - 1)
                    new_set.pop(idx)
            else:
                # Swap: remove one, add one
                if len(current_set) > 1:
                    idx = random.randint(0, len(current_set) - 1)
                    removed = new_set.pop(idx)
                    available = [x for x in all_numbers if x not in current_set]
                    if available:
                        x = random.choice(available)
                        new_set.append(x)
            
            if len(new_set) == len(current_set) and sorted(new_set) == sorted(current_set):
                continue
            
            new_violations = count_violations(new_set)
            new_size = len(new_set)
            old_size = len(current_set)
            
            # Objective: we want to maximize size and minimize violations
            # Score = size - weight * violations
            weight = 5.0  # Penalty weight for violations
            
            old_score = old_size - weight * old_violations
            new_score = new_size - weight * new_violations
            delta = new_score - old_score
            
            # Accept if improvement, or with probability exp(delta/T) if worse
            if delta >= 0:
                accept = True
            else:
                accept = random.random() < np.exp(delta / T)
            
            if accept:
                current_set = new_set
                current_violations = new_violations
                
                # Track best
                if current_violations == 0 and len(current_set) > best_size:
                    best_set = list(current_set)
                    best_size = len(current_set)
                    best_violations = 0
                elif current_violations < best_violations:
                    best_set = list(current_set)
                    best_violations = current_violations
                    best_size = len(current_set)
            
            # Cool down
            T *= cooling_rate
        
        # Return the best valid set found, or the least-violating set
        if best_violations == 0:
            return best_set
        else:
            # Try to repair: remove elements causing violations
            repaired = list(best_set)
            violations = count_violations(repaired)
            while violations > 0 and len(repaired) > 1:
                # Find an element causing a violation and remove it
                for a in repaired:
                    S_excl_a = [x for x in repaired if x != a]
                    if len(S_excl_a) < 2:
                        continue
                    dp_excl = compute_dp(S_excl_a)
                    m = len(S_excl_a)
                    for k in range(2, m + 1):
                        target_sum = k * a
                        if target_sum > sum(S_excl_a):
                            break
                        if (dp_excl[k] >> target_sum) & 1:
                            repaired.remove(a)
                            violations = count_violations(repaired)
                            break
                    if violations == 0:
                        break
            return repaired

    # Strategy: 
    # 1. Generate candidate orders.
    # 2. The "Prime First" heuristic is strong. Primes are less likely to be averages.
    # 3. We will try a few variations:
    #    a. Primes first, then composites in descending order.
    #    b. Primes first, then composites in ascending order.
    #    c. Randomized with prime bias.
    
    candidates = list(range(1, 513))
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    orders = []
    
    # Order 1: Primes descending, then composites descending
    orders.append(sorted(primes, reverse=True) + sorted(composites, reverse=True))
    
    # Order 2: Primes ascending, then composites ascending
    orders.append(sorted(primes) + sorted(composites))
    
    # Order 3: Primes descending, then composites ascending
    orders.append(sorted(primes, reverse=True) + sorted(composites))
    
    # Order 4: Interleaved? Or just random with prime bias
    # Let's do a few random shuffles but ensure primes are in the first half?
    # Actually, just standard random shuffles of the full list, but we run many.
    for _ in range(5):
        r = list(candidates)
        random.shuffle(r)
        orders.append(r)
        
    best_size = 0
    best_result = []
    
    for order in orders:
        if time.time() > deadline - 20:
            break
            
        res = greedy_construct(order)
        if len(res) > best_size:
            best_size = len(res)
            best_result = res
            
        # If we hit the max, we can stop early? 
        # But maybe a different order allows 48 where this one allows 47? 
        # The task says "Maximize cardinality". 48 is the hard cap.
        if best_size == 48:
            break

    # Simulated Annealing Phase
    if best_size < 48 and time.time() < deadline - 15:
        # Use the best greedy result as starting point
        sa_result = simulated_annealing(best_result, max_iter=3000)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_result = sa_result
    
    # Local Search / Improvement Phase
    # If we didn't reach 48, try to swap elements to allow adding more.
    if best_size < 48 and time.time() < deadline - 10:
        current_best = best_result[:]
        unused = [x for x in range(1, 513) if x not in current_best]
        
        # Try to add more elements directly first
        random.shuffle(unused)
        for x in unused:
            if time.time() > deadline - 2:
                break
            if len(current_best) >= 48:
                break
            if is_valid_addition(current_best, x):
                current_best.append(x)
        
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_result = current_best
        
        # If still not 48, try swaps.
        # This is expensive. Only do a limited number of swaps.
        if len(current_best) < 48:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try swapping one element in current_best with one in unused
                # We want to see if this swap allows adding a new element.
                # To speed up, we can just try random swaps and check if the set size can increase by 1 afterwards.
                
                # Pick a random element to remove
                idx_to_remove = random.randint(0, len(current_best) - 1)
                elem_to_remove = current_best[idx_to_remove]
                
                # Try a few random candidates to replace it
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    if not unused:
                        break
                    elem_to_add = random.choice(unused)
                    
                    # Test swap: Remove elem_to_remove, Add elem_to_add
                    # New set: (current_best - {elem_to_remove}) + {elem_to_add}
                    # Check if this new set is valid.
                    # If valid, can we add ANOTHER element?
                    # Or does the swap itself fix a conflict that prevented addition?
                    
                    # Actually, a simpler local search: 
                    # Just try to add elements to the current set. If stuck, 
                    # maybe removing a "bad" element allows adding two "good" ones?
                    
                    # Let's try a specific heuristic:
                    # Identify elements that are "means" of other subsets? 
                    # We don't track that. 
                    
                    # Let's just try random swaps to see if we can get a set that allows more additions.
                    new_set = [x for x in current_best if x != elem_to_remove]
                    if elem_to_add in new_set:
                        continue
                    new_set.append(elem_to_add)
                    
                    # Is new_set valid?
                    if not is_valid_set(new_set):
                        continue
                    
                    # If valid, try to add more elements to new_set
                    potential_set = new_set[:]
                    remaining_unused = [x for x in unused if x != elem_to_add and x not in potential_set]
                    random.shuffle(remaining_unused)
                    
                    for x in remaining_unused:
                        if len(potential_set) >= 48:
                            break
                        if is_valid_addition(potential_set, x):
                            potential_set.append(x)
                    
                    if len(potential_set) > len(current_best):
                        current_best = potential_set
                        best_size = len(current_best)
                        best_result = current_best
                        improved = True
                        # Update unused
                        unused = [x for x in range(1, 513) if x not in current_best]
                        break

    return best_result
# EVOLVE-BLOCK-END
