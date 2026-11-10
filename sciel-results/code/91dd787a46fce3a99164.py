import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def is_prime(x):
        if x < 2: return False
        if x < 4: return True
        if x % 2 == 0 or x % 3 == 0: return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def is_valid_candidate(x, S, dp, max_sum):
        """
        Check if x can be added to S.
        S: list of currently selected integers.
        dp: list of integers, where dp[k] is a bitmask of possible sums using exactly k elements from S.
        """
        n = len(S)
        
        # 1. Check if x is the mean of any subset of S of size k >= 2.
        # sum(subset) = k * x
        for k in range(2, n + 1):
            target = k * x
            if target < 0 or target > max_sum:
                continue
            # Check if bit 'target' is set in dp[k]
            if (dp[k] >> target) & 1:
                return False
        
        # 2. Check if any a in S becomes the mean of a subset of S U {x} of size >= 2.
        # A subset T' of S U {x} containing x can be written as T U {x} where T is a subset of S.
        # If x is not in the subset, it was already a valid mean (or not) in S.
        # We only need to check subsets that include x.
        # Let T' = T U {x}. Size |T'| = |T| + 1 = k + 1, where k = |T| >= 1 (since T' size >= 2).
        # Mean condition: sum(T) + x = (k + 1) * a
        # sum(T) = (k + 1) * a - x
        # We need to check if this sum is achievable by a subset T of S of size k.
        
        for a in S:
            for k in range(1, n + 1):
                target = (k + 1) * a - x
                if target < 0 or target > max_sum:
                    continue
                # Check if bit 'target' is set in dp[k]
                if (dp[k] >> target) & 1:
                    return False
        
        return True

    def try_ordering(order, max_sum=24576):
        S = []
        # dp[k] is bitmask of sums using k elements from S.
        # Initially S is empty. dp[0] = 1 (sum 0).
        dp = [0] * (49) 
        dp[0] = 1
        
        for x in order:
            if len(S) >= 48:
                break
            if time.time() > deadline - 5:
                break
            valid = is_valid_candidate(x, S, dp, max_sum)
            if valid:
                S.append(x)
                # Update DP: new subsets containing x
                # Iterate backwards to avoid using x multiple times in one step
                for k in range(len(S) - 1, 0, -1):
                    dp[k+1] |= (dp[k] << x)
                dp[1] |= (dp[0] << x)
                        
        return S

    def verify_set(S):
        """Verify that no element in S is the mean of any subset of at least 2 other elements."""
        n = len(S)
        if n < 3:
            return True
        
        S_set = set(S)
        max_sum = sum(S)
        
        # Build DP for all subset sizes
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for x in S:
            for k in range(n - 1, 0, -1):
                dp[k+1] |= (dp[k] << x)
            dp[1] |= (dp[0] << x)
        
        # For each element a in S, check if a is the mean of any subset of S \ {a}
        for a in S:
            others = [x for x in S if x != a]
            m = len(others)
            if m < 2:
                continue
            
            # Build DP for subsets of others
            dp_others = [0] * (m + 1)
            dp_others[0] = 1
            for x in others:
                for k in range(m - 1, 0, -1):
                    dp_others[k+1] |= (dp_others[k] << x)
                dp_others[1] |= (dp_others[0] << x)
            
            # Check if a is the mean of any subset of size k >= 2
            for k in range(2, m + 1):
                target = k * a
                if target < 0 or target > max_sum:
                    continue
                if (dp_others[k] >> target) & 1:
                    return False
        
        return True

    def local_search(S, max_sum=24576, max_iterations=50):
        """Try to improve the set by adding elements or swapping."""
        if not S:
            return S
        
        S = S[:]
        S_set = set(S)
        
        # Rebuild dp for current S
        dp = [0] * (49)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, 0, -1):
                dp[k+1] |= (dp[k] << x)
            dp[1] |= (dp[0] << x)
        
        nums = list(range(1, 513))
        candidates = [x for x in nums if x not in S_set]
        rng = random.Random(hash(tuple(sorted(S))) % 10000)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try to add a new element
            if len(S) < 48:
                for x in rng.sample(candidates, min(30, len(candidates))):
                    if len(S) >= 48:
                        break
                    if is_valid_candidate(x, S, dp, max_sum):
                        S.append(x)
                        S_set.add(x)
                        # Update dp
                        for k in range(len(S) - 1, 0, -1):
                            dp[k+1] |= (dp[k] << x)
                        dp[1] |= (dp[0] << x)
                        improved = True
                        candidates = [y for y in nums if y not in S_set]
            
            # Try to swap: remove one, add another
            if len(S) < 48:
                continue
            
            for i in range(len(S)):
                if time.time() > deadline - 5:
                    break
                removed = S[i]
                # Try adding candidates in place of removed
                for x in rng.sample(candidates, min(15, len(candidates))):
                    if x == removed:
                        continue
                    
                    # Build S_temp without S[i] but with x
                    S_temp = S[:i] + S[i+1:] + [x]
                    
                    # Check if x is valid with S without S[i]
                    S_without = S[:i] + S[i+1:]
                    n_without = len(S_without)
                    dp_without = [0] * (49)
                    dp_without[0] = 1
                    for y in S_without:
                        for k in range(n_without - 1, 0, -1):
                            dp_without[k+1] |= (dp_without[k] << y)
                        dp_without[1] |= (dp_without[0] << y)
                    
                    if is_valid_candidate(x, S_without, dp_without, max_sum):
                        # Verify the full set is valid
                        if verify_set(S_temp):
                            S = S_temp
                            S_set = set(S_temp)
                            # Rebuild dp
                            dp = [0] * (49)
                            dp[0] = 1
                            for y in S:
                                for k in range(len(S) - 1, 0, -1):
                                    dp[k+1] |= (dp[k] << y)
                                dp[1] |= (dp[0] << y)
                            candidates = [y for y in nums if y not in S_set]
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return S

    # Generate orderings
    nums = list(range(1, 513))
    
    # 1. Ascending
    ord_asc = nums[:]
    
    # 2. Descending
    ord_desc = nums[::-1]
    
    # 3. Primes first
    primes = [x for x in nums if is_prime(x)]
    non_primes = [x for x in nums if not is_prime(x)]
    ord_prime = primes + non_primes
    
    # 4. Alternating small/large
    alt = []
    for i in range(256):
        if i+1 <= 512: alt.append(i+1)
        if 512-i >= 1: alt.append(512-i)
    ord_alt = alt
    
    # 5. Powers of 2 first
    powers = [2**i for i in range(10) if 2**i <= 512]
    others = [x for x in nums if x not in set(powers)]
    ord_powers = powers + others
    
    # 6. Fibonacci-like
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= 512:
        fibs.append(fibs[-1] + fibs[-2])
    fibs = list(dict.fromkeys(fibs))  # Remove duplicates, preserve order
    others_fib = [x for x in nums if x not in set(fibs)]
    ord_fib = fibs + others_fib
    
    # 7. Random shuffles
    rng = random.Random(42)
    ords = [ord_asc, ord_desc, ord_prime, ord_alt, ord_powers, ord_fib]
    for _ in range(25):
        o = nums[:]
        rng.shuffle(o)
        ords.append(o)

    best_set = []
    best_len = 0
    
    for o in ords:
        if time.time() > deadline - 10:
            break
        s = try_ordering(o)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
        
        # Try local search to improve
        if len(s) > 0 and len(s) < 48:
            s_improved = local_search(s)
            if len(s_improved) > best_len:
                best_set = s_improved
                best_len = len(s_improved)
        
        if best_len == 48:
            break
            
    # Final verification
    if best_set and not verify_set(best_set):
        # Fall back to a simpler approach if verification fails
        S = try_ordering(ord_asc)
        if verify_set(S):
            best_set = S
    
    return best_set
