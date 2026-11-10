import time
import random

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2."""
        n = len(others)
        if n < 2:
            return False
        
        max_sum = sum(others)
        # We need subset sum = k * element for k >= 2
        # Max k is n. Max target sum is n * element.
        # But also bounded by max_sum.
        
        # Bitset DP: dp[j] is a bitset where bit s is set if sum s is achievable with j elements
        dp = [0] * (n + 1)
        dp[0] = 1
        
        # Optimization: Process elements in a way that might fill the bitset faster? 
        # Or just standard.
        for val in others:
            for j in range(n, 0, -1):
                if dp[j - 1]:
                    dp[j] |= (dp[j - 1] << val)
        
        for k in range(2, n + 1):
            target_sum = k * element
            if target_sum > max_sum:
                break
            if (dp[k] >> target_sum) & 1:
                return True
        return False

    def can_add(element, current_set):
        """Check if element can be added to current_set."""
        if not current_set:
            return True
        
        # Check 1: Is 'element' the mean of any subset of 'current_set'?
        if is_mean_of_subset(element, current_set):
            return False
        
        # Check 2: Does adding 'element' make any existing element 'a' a mean?
        # If 'a' becomes a mean, the subset must include 'element'.
        # Let T be a subset of current_set \ {a}.
        # a = mean(T U {element}) => sum(T) + element = (|T| + 1) * a
        # sum(T) = (|T| + 1) * a - element
        # Let k = |T| + 1. Then sum(T) = k * a - element.
        # T has size k - 1.
        
        n = len(current_set)
        for i, a in enumerate(current_set):
            others_without_a = current_set[:i] + current_set[i+1:]
            if len(others_without_a) < 1:
                continue
            
            max_sum_others = sum(others_without_a)
            
            # We need to check if there exists a subset T of others_without_a
            # of size j (where j = k-1 >= 1) such that sum(T) = (j+1)*a - element
            # Note: k >= 2 => j >= 1.
            
            # Bitset DP for others_without_a
            m = len(others_without_a)
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for val in others_without_a:
                for j in range(m, 0, -1):
                    if dp[j - 1]:
                        dp[j] |= (dp[j - 1] << val)
            
            for j in range(1, m + 1):
                target_sum = (j + 1) * a - element
                if target_sum < 0 or target_sum > max_sum_others:
                    continue
                if (dp[j] >> target_sum) & 1:
                    return False
        
        return True

    def is_valid_set(s):
        """Verify the entire set satisfies the constraint."""
        n = len(s)
        for i in range(n):
            element = s[i]
            others = s[:i] + s[i+1:]
            if is_mean_of_subset(element, others):
                return False
        return True

    # Generate diverse orderings
    all_nums = list(range(1, 513))
    
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0: return False
        return True

    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    
    # Powers of 2
    p2 = [2**i for i in range(1, 10) if 2**i <= 512]
    non_p2 = [x for x in range(1, 513) if x not in set(p2)]
    
    # Alternating
    alt = []
    for i in range(256):
        alt.append(i + 1)
        alt.append(512 - i)
        
    # Random
    rng = random.Random(42)
    rnd = all_nums[:]
    rng.shuffle(rnd)
    
    # Modulo based
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
            if can_add(x, selected):
                selected.append(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) == 48:
                break

    # If we didn't reach 48, try local search (swaps) to improve
    if len(best_set) < 48 and time.time() < deadline - 10:
        current = best_set[:]
        current_set = set(current)
        remaining = [x for x in range(1, 513) if x not in current_set]
        
        improved = False
        # Try to add more if possible (should have been caught by greedy, but maybe order matters)
        for x in remaining:
            if time.time() > deadline - 10:
                break
            if len(current) >= 48:
                break
            if can_add(x, current):
                current.append(x)
                current_set.add(x)
                improved = True
        if improved:
            best_set = current
            if len(best_set) == 48:
                return best_set

        # Try swaps: remove one, add two? Or remove one, add one?
        # Since we are maximizing size, we want net gain.
        # If we remove 1 and add 2, size increases by 1.
        # This is expensive. Let's just try to add more first.
        
        # If still < 48, try random restarts with different seeds
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 5:
                break
            rng = random.Random(seed)
            order = list(range(1, 513))
            rng.shuffle(order)
            
            selected = []
            for x in order:
                if time.time() > deadline - 5:
                    break
                if len(selected) >= 48:
                    break
                if can_add(x, selected):
                    selected.append(x)
                    
            if len(selected) > len(best_set):
                best_set = selected
                if len(best_set) == 48:
                    break

    # Final check
    if best_set and not is_valid_set(best_set):
        # Fallback to a known valid small set if verification fails (should not happen)
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
