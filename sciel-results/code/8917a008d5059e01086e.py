import time
import random
import numpy as np

def construct():
    deadline = time.time() + 88
    
    # Precompute useful data
    all_nums = list(range(1, 513))
    
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True

    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    
    p2 = [2**i for i in range(1, 10) if 2**i <= 512]
    non_p2 = [x for x in range(1, 513) if x not in set(p2)]
    
    # Generate more diverse orderings
    orderings = []
    
    # Original orderings
    orderings.append(p2 + non_p2)
    orderings.append(primes + non_primes)
    
    alt = []
    for i in range(256):
        alt.append(i + 1)
        alt.append(512 - i)
    orderings.append(alt)
    
    orderings.append(list(range(1, 513)))
    orderings.append(list(range(512, 0, -1)))
    
    # Modular orderings for different moduli
    for mod in [3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31]:
        mod_order = []
        for r in range(mod):
            mod_order.extend([x for x in range(1, 513) if x % mod == r])
        orderings.append(mod_order)
    
    # Fibonacci-like ordering
    fib = [1, 1]
    while fib[-1] <= 512:
        fib.append(fib[-1] + fib[-2])
    fib_order = list(dict.fromkeys(fib))  # Remove duplicates, preserve order
    remaining_fib = [x for x in range(1, 513) if x not in set(fib_order)]
    orderings.append(fib_order + remaining_fib)
    
    # Random orderings with different seeds
    for seed in [42, 123, 456, 789, 1011, 2023, 3141, 2718, 1618, 9876]:
        rng = random.Random(seed)
        rnd = all_nums[:]
        rng.shuffle(rnd)
        orderings.append(rnd)
    
    # Interleaved even/odd
    evens = [x for x in range(1, 513) if x % 2 == 0]
    odds = [x for x in range(1, 513) if x % 2 == 1]
    interleaved = []
    for i in range(max(len(evens), len(odds))):
        if i < len(evens):
            interleaved.append(evens[i])
        if i < len(odds):
            interleaved.append(odds[i])
    orderings.append(interleaved)

    def can_add_fast(element, others):
        """
        Check if element is the mean of any subset of 'others' (size >= 2).
        Uses bitset DP with optimizations.
        """
        n = len(others)
        if n < 2:
            return True  # Can always add if not enough others to form mean
        
        max_sum = sum(others)
        if max_sum < 2 * element:
            return True
        
        # Precompute sorted others for min/max sum bounds
        others_sorted = sorted(others)
        
        # dp[j] is a bitset where bit s is set if sum s is achievable with j elements
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for val in others:
            for j in range(n, 0, -1):
                if dp[j - 1]:
                    dp[j] |= (dp[j - 1] << val)
        
        # Check if any k >= 2 has target sum k*element
        # With early termination using min/max possible sums
        for k in range(2, n + 1):
            target_sum = k * element
            
            # Early termination: if target_sum > max_sum, no larger k will work
            if target_sum > max_sum:
                break
            
            # Min sum of k elements is sum of smallest k
            min_sum_k = sum(others_sorted[:k])
            # Max sum of k elements is sum of largest k
            max_sum_k = sum(others_sorted[-k:])
            
            # If target_sum is outside achievable range, skip
            if target_sum < min_sum_k or target_sum > max_sum_k:
                continue
            
            if (dp[k] >> target_sum) & 1:
                return False
        return True

    def is_valid_set(s):
        """
        Verify the entire set satisfies the constraint:
        No element is the mean of any subset of at least two OTHER elements.
        """
        n = len(s)
        if n < 3:
            return True
        for i in range(n):
            element = s[i]
            others = s[:i] + s[i+1:]
            if not can_add_fast(element, others):
                return False
        return True

    def can_add_with_context(element, current_set):
        """
        Check if 'element' can be added to 'current_set'.
        1. 'element' is not mean of any subset of current_set.
        2. No existing element 'a' in current_set becomes mean of a subset involving 'element'.
        """
        if not current_set:
            return True
        
        # Check 1: Is 'element' the mean of any subset of 'current_set'?
        if not can_add_fast(element, current_set):
            return False
        
        # Check 2: Does adding 'element' make any existing element 'a' a mean?
        # If 'a' becomes a mean, the subset MUST include 'element'.
        # Let T be a subset of current_set \ {a}.
        # a = mean(T U {element}) => sum(T) + element = (|T| + 1) * a
        # sum(T) = (|T| + 1) * a - element
        # Let k = |T| + 1. Then sum(T) = k * a - element.
        # T has size k - 1.
        # Since the original subset for 'a' must have size >= 2, and it includes 'element',
        # the total size is k. So k >= 2 => |T| >= 1.
        
        n = len(current_set)
        current_set_sorted = sorted(current_set)
        
        # Precompute prefix sums for quick min/max sum calculations
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + current_set_sorted[i]
        
        # We iterate through each 'a' in current_set.
        for i, a in enumerate(current_set):
            others_without_a = current_set[:i] + current_set[i+1:]
            if len(others_without_a) < 1:
                continue
            
            # We need to check if there exists a subset T of others_without_a
            # of size j (where j = k-1 >= 1) such that sum(T) = (j+1)*a - element
            
            m = len(others_without_a)
            others_without_a_sorted = sorted(others_without_a)
            
            # Precompute prefix sums for others_without_a
            prefix_others = [0] * (m + 1)
            for idx in range(m):
                prefix_others[idx + 1] = prefix_others[idx] + others_without_a_sorted[idx]
            
            # Bitset DP for others_without_a
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for val in others_without_a:
                for j in range(m, 0, -1):
                    if dp[j - 1]:
                        dp[j] |= (dp[j - 1] << val)
            
            # Check for any valid j
            for j in range(1, m + 1):
                target_sum = (j + 1) * a - element
                # target_sum must be achievable by j elements from others_without_a
                # Also target_sum must be >= 0 and <= sum(others_without_a)
                
                if target_sum < 0:
                    continue
                
                # Use precomputed prefix sums for min/max bounds
                min_sum_j = prefix_others[j]  # Sum of smallest j
                max_sum_j = prefix_others[m] - prefix_others[m - j]  # Sum of largest j
                
                if target_sum < min_sum_j or target_sum > max_sum_j:
                    continue
                
                if (dp[j] >> target_sum) & 1:
                    return False
        
        return True

    # --- Search Strategies ---

    # Strategy 1: Greedy with diverse orderings
    all_nums = list(range(1, 513))
    
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0: return False
        return True

    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    
    p2 = [2**i for i in range(1, 10) if 2**i <= 512]
    non_p2 = [x for x in range(1, 513) if x not in set(p2)]
    
    alt = []
    for i in range(256):
        alt.append(i + 1)
        alt.append(512 - i)
        
    rng = random.Random(42)
    rnd = all_nums[:]
    rng.shuffle(rnd)
    
    mod5 = []
    for r in range(5):
        mod5.extend([x for x in range(1, 513) if x % 5 == r])
        
    orderings = [
        p2 + non_p2,
        primes + non_primes,
        alt,
        list(range(1, 513)),
        list(range(512, 0, -1)),
        mod5,
        rnd
    ]

    best_set = []
    
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
            
        selected = []
        for x in ordering:
            if time.time() > deadline - 5:
                break
            if len(selected) >= 48:
                break
            if can_add_with_context(x, selected):
                selected.append(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) == 48:
                return best_set

    # Strategy 2: Local Search / Hill Climbing
    # Try to improve the size of the set by swapping or adding.
    # Since we want to maximize size, we primarily want to ADD elements.
    # If we are stuck at size K < 48, we try to find an element x not in the set
    # such that adding it doesn't violate constraints.
    # If we can't add any single element, we might try removing one element and adding two?
    # Or removing one and adding one (if the new one allows more additions later)?
    # Let's stick to: 
    # 1. Try to add any remaining element.
    # 2. If that fails, try random restarts with more randomness.

    if len(best_set) < 48:
        current = best_set[:]
        current_set = set(current)
        remaining = [x for x in range(1, 513) if x not in current_set]
        
        # Try to add more if possible (should have been caught by greedy, but maybe order matters)
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            for x in remaining:
                if time.time() > deadline - 10:
                    break
                if len(current) >= 48:
                    break
                if can_add_with_context(x, current):
                    current.append(x)
                    current_set.add(x)
                    remaining.remove(x)
                    improved = True
                    break # Restart loop with new state
            if len(current) > len(best_set):
                best_set = current[:]
                if len(best_set) == 48:
                    return best_set

    # Strategy 3: If still < 48, try more random restarts with different seeds
    # and also try some biased orderings
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed * 1000 + 1)
        order = list(range(1, 513))
        rng.shuffle(order)
        
        selected = []
        for x in order:
            if time.time() > deadline - 5:
                break
            if len(selected) >= 48:
                break
            if can_add_with_context(x, selected):
                selected.append(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) == 48:
                return best_set
    
    # Strategy 4: Try biased orderings that favor numbers less likely to be means
    # Numbers that are prime or powers of 2 might be "safer"
    biased_orders = [
        # Primes first
        primes + non_primes,
        # Powers of 2 first
        p2 + non_p2,
        # Numbers coprime to many others first (heuristic)
        sorted(range(1, 513), key=lambda x: sum(1 for y in range(1, 513) if y != x and x % y == 0)),
    ]
    
    for order in biased_orders:
        if time.time() > deadline - 5:
            break
        
        selected = []
        for x in order:
            if time.time() > deadline - 5:
                break
            if len(selected) >= 48:
                break
            if can_add_with_context(x, selected):
                selected.append(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) == 48:
                return best_set

    # Final check
    if best_set and not is_valid_set(best_set):
        # Fallback to a known valid small set if verification fails (should not happen)
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
