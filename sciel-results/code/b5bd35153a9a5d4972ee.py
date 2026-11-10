# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    n_max = 512
    
    # We'll work with a boolean array: kept[i] means (i+1) is in our set
    # Start with all kept, then remove elements to satisfy constraints
    # Objective: minimize number of removals (maximize |kept|)
    
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Precompute primes for bias
    primes = set(x for x in range(1, n_max + 1) if is_prime(x))
    
    # Fast incremental validity check
    # We maintain:
    #   kept: set of kept elements
    #   For each element a in kept, we track whether it's a "bad" mean
    # But tracking all means is expensive. Instead, we use a different approach:
    # 
    # Key insight: a is the mean of subset T (|T|>=2, T subset of kept\{a}) iff
    #   sum(T) = k*a for some k>=2
    # 
    # For incremental checks when adding x to kept:
    #   1. Is x the mean of any subset of kept (old) of size >=2?
    #   2. For each a in kept (old), is a the mean of any subset of (kept_old ∪ {x})\{a} of size >=2?
    #      This subset must contain x (otherwise it was already checked when a was added).
    #      So: sum(T) + x = (|T|+1)*a where T subset of kept_old\{a}
    #      => sum(T) = (|T|+1)*a - x
    #
    # For removing x from kept:
    #   Only violations involving x can be fixed. No new violations are created.
    #   So removal is always "safe" in terms of not creating new violations.
    #   But we need to check if the remaining set is valid (it was valid before, 
    #   and removing elements can only fix violations, not create new ones).
    #   Actually, that's not quite right. Removing x might make some a no longer 
    #   a mean, which is fine. But it can't make some a a mean that wasn't before.
    #   So removal is always valid if the set was valid before.
    
    # Given the complexity, let's use a simpler but effective approach:
    # Maintain the set and use a fast check for single-element additions.
    # For the annealer, we'll use:
    #   - State: set of kept elements (as a Python set for O(1) membership)
    #   - Move: add one element, remove one element, or swap
    #   - Cost: number of violations (0 = valid)
    #   - We compute cost incrementally
    
    # For efficiency, we'll precompute subset sums for small subsets
    # Actually, let's use the following approach:
    # 
    # We'll maintain a list of "violation counts" per element.
    # But this is complex. Let's use a pragmatic approach:
    # 
    # Since n_max=512 and we want ~48 elements, the set is sparse.
    # We can afford to do a moderately expensive check for each move.
    # 
    # Let's use the following fast check:
    # When adding x to set S:
    #   Check 1: x is not the mean of any subset of S of size >= 2
    #   Check 2: For each a in S, a is not the mean of any subset of (S ∪ {x})\{a} of size >= 2
    #            that includes x.
    #
    # For Check 1, we need: does S have a subset of size k>=2 with sum = k*x?
    # For Check 2, for each a in S: does S\{a} have a subset of size j>=1 with sum = (j+1)*a - x?
    
    # Let's implement a fast subset-sum check using bitsets for small sets
    # Since |S| <= ~50, we can use Python integers as bitsets
    
    def subset_sums_bitset(nums):
        """Return list dp where dp[k] is a bitset: bit s is 1 if subset of size k sums to s"""
        n = len(nums)
        if n == 0:
            return [1]
        max_sum = sum(nums)
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in nums:
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        return dp
    
    def has_subset_sum(nums, k, target):
        """Check if there's a subset of nums of size exactly k with sum = target"""
        if k < 0 or k > len(nums):
            return False
        if target < 0 or target > sum(nums):
            return False
        if k == 0:
            return target == 0
        dp = subset_sums_bitset(nums)
        return (dp[k] >> target) & 1
    
    # For the annealer, we'll use a simpler cost function:
    # cost(S) = number of elements a in S that are the mean of some subset of S\{a} of size >= 2
    # We'll compute this from scratch for each state, but optimize with early termination
    
    def count_violations(S):
        """Count how many elements in S are means of subsets of S\{a} of size >= 2"""
        if len(S) < 3:
            return 0
        S_list = list(S)
        n = len(S_list)
        violations = 0
        for i, a in enumerate(S_list):
            S_excl = [x for x in S_list if x != a]
            m = len(S_excl)
            if m < 2:
                continue
            dp = subset_sums_bitset(S_excl)
            found = False
            for k in range(2, m + 1):
                target = k * a
                if target > sum(S_excl):
                    break
                if (dp[k] >> target) & 1:
                    found = True
                    break
            if found:
                violations += 1
        return violations
    
    # Initial state: start with a small valid set and grow
    # Let's start with primes in a random order and greedily add
    
    def greedy_init():
        S = set()
        # Try multiple random orders, biased toward primes
        best = set()
        for trial in range(20):
            if time.time() > deadline - 30:
                break
            candidates = list(range(1, n_max + 1))
            # Bias: put primes first
            random.shuffle(candidates)
            # Sort by prime bias with some randomness
            candidates.sort(key=lambda x: (0 if x in primes else 1, random.random()))
            S_trial = set()
            for x in candidates:
                if len(S_trial) >= 48:
                    break
                # Check if x can be added
                S_list = list(S_trial)
                # Check 1: x not mean of subset of S_trial of size >= 2
                valid = True
                if len(S_list) >= 2:
                    dp = subset_sums_bitset(S_list)
                    for k in range(2, len(S_list) + 1):
                        target = k * x
                        if target > sum(S_list):
                            break
                        if (dp[k] >> target) & 1:
                            valid = False
                            break
                if not valid:
                    continue
                # Check 2: for each a in S_trial, a not mean of subset of (S_trial ∪ {x})\{a} of size >= 2
                for a in S_trial:
                    S_excl_a = [y for y in S_list if y != a]
                    if len(S_excl_a) < 1:
                        continue
                    dp_excl = subset_sums_bitset(S_excl_a)
                    for j in range(1, len(S_excl_a) + 1):
                        target = (j + 1) * a - x
                        if target < 0:
                            continue
                        if target > sum(S_excl_a):
                            break
                        if (dp_excl[j] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    S_trial.add(x)
            if len(S_trial) > len(best):
                best = S_trial
            if len(best) >= 48:
                break
        return best
    
    # Simulated annealing on the set
    def simulated_annealing(initial_set):
        S = set(initial_set)
        if len(S) >= 48:
            return S
        
        best_S = set(S)
        best_cost = count_violations(S)
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.1
        iterations = 50000
        # But we're time-limited, so we'll iterate until deadline
        
        # For moves, we'll try:
        # 1. Add a random element from [1..512]\S
        # 2. Remove a random element from S
        # 3. Swap: remove one from S, add one from [1..512]\S
        
        all_elements = list(range(1, n_max + 1))
        
        iter_count = 0
        while time.time() < deadline - 5:
            iter_count += 1
            T = T_start * (T_end / T_start) ** (iter_count / iterations)
            if T < T_end:
                T = T_end
            
            # Choose a move
            move_type = random.random()
            if move_type < 0.4:
                # Add
                if len(S) >= 48:
                    continue
                candidates = [x for x in all_elements if x not in S]
                if not candidates:
                    continue
                x_add = random.choice(candidates)
                new_S = set(S)
                new_S.add(x_add)
            elif move_type < 0.7:
                # Remove
                if len(S) <= 1:
                    continue
                x_rem = random.choice(list(S))
                new_S = set(S)
                new_S.discard(x_rem)
            else:
                # Swap
                if len(S) <= 1:
                    continue
                x_rem = random.choice(list(S))
                candidates = [x for x in all_elements if x not in S]
                if not candidates:
                    continue
                x_add = random.choice(candidates)
                new_S = set(S)
                new_S.discard(x_rem)
                new_S.add(x_add)
            
            # Compute cost
            new_cost = count_violations(new_S)
            delta_cost = new_cost - best_cost
            
            # Acceptance
            if delta_cost <= 0:
                S = new_S
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_S = set(S)
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta_cost / max(T, 1e-9)):
                    S = new_S
            
            # Early termination if we hit 48
            if len(S) >= 48 and count_violations(S) == 0:
                return S
        
        return best_S
    
    # Run
    initial = greedy_init()
    result = simulated_annealing(initial)
    
    # Final greedy fill
    if len(result) < 48:
        candidates = [x for x in range(1, n_max + 1) if x not in result]
        random.shuffle(candidates)
        for x in candidates:
            if len(result) >= 48:
                break
            if time.time() > deadline - 2:
                break
            # Check if x can be added
            S_list = list(result)
            valid = True
            if len(S_list) >= 2:
                dp = subset_sums_bitset(S_list)
                for k in range(2, len(S_list) + 1):
                    target = k * x
                    if target > sum(S_list):
                        break
                    if (dp[k] >> target) & 1:
                        valid = False
                        break
            if not valid:
                continue
            for a in S_list:
                S_excl_a = [y for y in S_list if y != a]
                if len(S_excl_a) < 1:
                    continue
                dp_excl = subset_sums_bitset(S_excl_a)
                for j in range(1, len(S_excl_a) + 1):
                    target = (j + 1) * a - x
                    if target < 0:
                        continue
                    if target > sum(S_excl_a):
                        break
                    if (dp_excl[j] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                result.add(x)
    
    return list(result)
# EVOLVE-BLOCK-END
