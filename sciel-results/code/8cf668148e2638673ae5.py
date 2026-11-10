import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 512
    MAX_SIZE = 48
    MAX_SUM = MAX_SIZE * N  # 48 * 512 = 24576

    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5)+1):
            if x % i == 0:
                return False
        return True

    def check_3term_ap_conflict(x, S):
        """
        Check if adding x to S creates a 3-term arithmetic progression.
        i.e., x = (a+b)/2, or a = (x+b)/2, or b = (x+a)/2 for a,b in S.
        This is a necessary condition for validity (subset size 2).
        """
        for a in S:
            # x is mean of a and b -> b = 2x - a
            b = 2 * x - a
            if b in S and b != x:
                return True
            # a is mean of x and b -> b = 2a - x
            b = 2 * a - x
            if b in S and b != x and b != a:
                return True
        return False

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
            if (dp[k] >> target) & 1:
                return False
        
        # 2. Check if any a in S becomes the mean of a subset of S U {x} of size >= 2.
        # Subsets containing x: T' = T U {x}, T subset of S.
        # Size |T'| = |T| + 1 = k + 1, where k = |T| >= 1.
        # Mean condition: sum(T) + x = (k + 1) * a
        # sum(T) = (k + 1) * a - x
        for a in S:
            for k in range(1, n + 1):
                target = (k + 1) * a - x
                if target < 0 or target > max_sum:
                    continue
                if (dp[k] >> target) & 1:
                    return False
        
        return True

    def try_ordering(order, max_sum=24576):
        S = []
        S_set = set()
        dp = [0] * (MAX_SIZE + 1) 
        dp[0] = 1
        
        for x in order:
            if len(S) >= MAX_SIZE:
                break
            if time.time() > deadline - 5:
                break
            
            # Quick reject: 3-term AP check
            if check_3term_ap_conflict(x, S_set):
                continue
                
            valid = is_valid_candidate(x, S, dp, max_sum)
            if valid:
                S.append(x)
                S_set.add(x)
                # Update DP
                for k in range(len(S) - 1, 0, -1):
                    dp[k+1] |= (dp[k] << x)
                dp[1] |= (dp[0] << x)
                        
        return S, S_set, dp

    def local_search(S, S_set, dp, max_sum=24576):
        """
        Improve the set by adding elements or swapping.
        """
        if not S:
            return S, S_set, dp
        
        S = S[:]
        S_set = set(S_set)
        best_len = len(S)
        
        nums = list(range(1, 513))
        candidates = [x for x in nums if x not in S_set]
        
        # Rebuild dp if not provided correctly or to be safe
        dp = [0] * (MAX_SIZE + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, 0, -1):
                dp[k+1] |= (dp[k] << x)
            dp[1] |= (dp[0] << x)

        improved = True
        while improved and len(S) < MAX_SIZE:
            improved = False
            if time.time() > deadline - 5:
                break
            
            # 1. Try to add
            # Sort candidates by some heuristic? Random is fine.
            random.shuffle(candidates)
            for x in candidates:
                if len(S) >= MAX_SIZE:
                    break
                if check_3term_ap_conflict(x, S_set):
                    continue
                if is_valid_candidate(x, S, dp, max_sum):
                    S.append(x)
                    S_set.add(x)
                    for k in range(len(S) - 1, 0, -1):
                        dp[k+1] |= (dp[k] << x)
                    dp[1] |= (dp[0] << x)
                    candidates.remove(x)
                    improved = True
                    best_len = len(S)
                    if best_len == MAX_SIZE:
                        return S, S_set, dp
                    break # Restart adding loop with new state
            
            # 2. Try to swap if no add found
            if len(S) < MAX_SIZE:
                continue
                
            # If we are at max size, we can still try swaps to potentially allow more?
            # No, max size is 48. If we are at 48, we are done.
            # The task says "Maximize cardinality. At most 48 integers".
            # So if we reach 48, we stop.
            
            if len(S) == MAX_SIZE:
                return S, S_set, dp

            # If we couldn't add, try swaps to unblock?
            # Swapping doesn't increase size. It only changes composition.
            # We only care about size. So if we can't add, we are stuck in this local optimum.
            # We should restart with a different ordering.
            
        return S, S_set, dp

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
    fibs = list(dict.fromkeys(fibs))
    others_fib = [x for x in nums if x not in set(fibs)]
    ord_fib = fibs + others_fib

    # 7. Random shuffles
    rng = random.Random(42)
    ords = [ord_asc, ord_desc, ord_prime, ord_alt, ord_powers, ord_fib]
    for _ in range(50):
        o = nums[:]
        rng.shuffle(o)
        ords.append(o)

    best_set = []
    best_len = 0
    
    for o in ords:
        if time.time() > deadline - 10:
            break
        
        s, s_set, dp = try_ordering(o)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
            
        # Try local search to improve
        if len(s) > 0 and len(s) < MAX_SIZE:
            s_improved, _, _ = local_search(s, s_set, dp)
            if len(s_improved) > best_len:
                best_set = s_improved
                best_len = len(s_improved)
        
        if best_len == MAX_SIZE:
            break
            
    return best_set
