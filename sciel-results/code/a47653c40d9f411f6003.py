# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 80  # Leave 10s margin for safety
    N_MAX = 48
    LIMIT = 512
    
    def compute_dp(nums):
        """
        Computes DP bitsets for subset sums by cardinality.
        dp[k] is an integer where bit s is 1 if a subset of size k sums to s.
        """
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

    def is_valid_set(S):
        """
        Checks if S is a valid set.
        Optimized by computing DP for S and then checking exclusions.
        """
        if len(S) < 3:
            return True
        
        n = len(S)
        S_sorted = sorted(S)
        
        # Fast check: If any element is a mean of a subset of the whole set 
        # that does NOT include the element itself, it's invalid.
        # We compute DP for the whole set.
        dp_full = compute_dp(S_sorted)
        
        # For each a in S, we need to check if a is the mean of a subset of S \ {a}.
        # This is equivalent to: 
        # For each a, for each k >= 2:
        #   Is there a subset of S \ {a} of size k with sum k*a?
        
        # To speed this up, we can try to compute dp_excl for each a.
        # But that is O(N * (N * Sum)).
        # Alternative: 
        # If S is valid, then for any a, and any subset T of S \ {a} with sum k*a, 
        # such T must not exist.
        
        # Let's iterate through elements and compute dp for S \ {a} only if necessary?
        # No, we must check all.
        
        # Optimization: 
        # If a is small, k*a is small.
        # If a is large, k*a is large.
        
        for i, a in enumerate(S_sorted):
            # Compute DP for S \ {a}
            S_excl = S_sorted[:i] + S_sorted[i+1:]
            if len(S_excl) < 2:
                continue
            
            dp_excl = compute_dp(S_excl)
            m = len(S_excl)
            max_sum_excl = sum(S_excl)
            
            for k in range(2, m + 1):
                target_sum = k * a
                if target_sum > max_sum_excl:
                    break
                if (dp_excl[k] >> target_sum) & 1:
                    return False
        return True

    def is_valid_addition(S, x):
        """
        Checks if x can be added to valid set S.
        S is sorted list.
        """
        if x in S:
            return False
        
        n = len(S)
        
        # 1. Check if x is mean of subset of S
        if n >= 2:
            dp_S = compute_dp(S)
            for k in range(2, n + 1):
                target_sum = k * x
                if target_sum > sum(S):
                    break
                if (dp_S[k] >> target_sum) & 1:
                    return False
        
        # 2. Check if any a in S is mean of subset of S U {x} containing x
        # Subset T U {x}, T subset of S \ {a}
        # sum(T) + x = (|T| + 1) * a
        # sum(T) = (|T| + 1) * a - x
        for i, a in enumerate(S):
            S_excl_a = S[:i] + S[i+1:]
            if not S_excl_a:
                continue
            dp_excl = compute_dp(S_excl_a)
            m = len(S_excl_a)
            max_sum_excl = sum(S_excl_a)
            for j in range(1, m + 1):
                target_sum = (j + 1) * a - x
                if target_sum < 0:
                    continue
                if target_sum > max_sum_excl:
                    break
                if (dp_excl[j] >> target_sum) & 1:
                    return False
        return True

    # --- Main Logic ---
    
    # 1. Greedy Construction
    # Use a heuristic order: Primes first, then others.
    candidates = list(range(1, LIMIT + 1))
    
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0: return False
        return True

    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Try multiple orders
    best_set = []
    best_size = 0
    
    orders = [
        sorted(primes, reverse=True) + sorted(composites, reverse=True),
        sorted(primes) + sorted(composites),
        sorted(primes, reverse=True) + sorted(composites),
        sorted(composites, reverse=True) + sorted(primes, reverse=True),
    ]
    
    # Add some random shuffles
    for _ in range(3):
        r = candidates[:]
        random.shuffle(r)
        orders.append(r)
        
    for order in orders:
        if time.time() > deadline - 10:
            break
            
        current = []
        for x in order:
            if time.time() > deadline - 5:
                break
            if len(current) >= N_MAX:
                break
            if is_valid_addition(current, x):
                current.append(x)
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
        if best_size == N_MAX:
            break

    # 2. Local Search / Swap Improvement
    if best_size < N_MAX:
        current_best = best_set[:]
        unused = [x for x in range(1, LIMIT + 1) if x not in current_best]
        
        # Try to add more elements first
        random.shuffle(unused)
        for x in unused:
            if time.time() > deadline - 2:
                break
            if len(current_best) >= N_MAX:
                break
            if is_valid_addition(current_best, x):
                current_best.append(x)
                unused.remove(x)
                
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = current_best[:]

        # Swap-based Local Search
        if best_size < N_MAX:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                # Try random swaps
                # To make it effective, we should try to swap an element that is "causing" conflicts
                # but we don't track that easily. Random swap is okay for now.
                
                if not current_best or not unused:
                    break
                    
                # Pick a random element to remove
                idx_to_remove = random.randint(0, len(current_best) - 1)
                elem_to_remove = current_best[idx_to_remove]
                
                # Try a few random candidates to replace it
                # We limit the number of attempts per iteration to avoid long stalls
                attempts = 0
                max_attempts = 20
                while attempts < max_attempts and time.time() < deadline - 2:
                    attempts += 1
                    if not unused:
                        break
                    idx_to_add = random.randint(0, len(unused) - 1)
                    elem_to_add = unused[idx_to_add]
                    
                    # Perform swap
                    new_set = current_best[:]
                    new_set[idx_to_remove] = elem_to_add
                    new_set.sort()
                    
                    # Check validity of new_set
                    # This is expensive. 
                    # Optimization: Check if the swap is "likely" to be valid?
                    # For now, just check.
                    if is_valid_set(new_set):
                        # If valid, can we add more?
                        potential_set = new_set[:]
                        remaining_unused = [x for x in unused if x != elem_to_add and x not in potential_set]
                        random.shuffle(remaining_unused)
                        
                        for x in remaining_unused:
                            if len(potential_set) >= N_MAX:
                                break
                            if is_valid_addition(potential_set, x):
                                potential_set.append(x)
                        
                        if len(potential_set) > len(current_best):
                            current_best = potential_set
                            best_size = len(current_best)
                            best_set = current_best[:]
                            improved = True
                            # Update unused
                            unused = [x for x in range(1, LIMIT + 1) if x not in current_best]
                            break
                        else:
                            # Swap is valid but didn't allow adding more immediately.
                            # Accept the swap anyway? It might open up future swaps.
                            # Let's accept it if it's valid, as it explores the space.
                            # But we must be careful not to loop.
                            # Let's only accept if it increases size or if we are stuck.
                            # Since we didn't increase size, we might just accept it to change the landscape.
                            # However, is_valid_set is expensive.
                            # Let's just accept the swap if it's valid.
                            current_best = new_set
                            # Update unused
                            unused = [x for x in range(1, LIMIT + 1) if x not in current_best]
                            improved = True
                            break

    return best_set
# EVOLVE-BLOCK-END
