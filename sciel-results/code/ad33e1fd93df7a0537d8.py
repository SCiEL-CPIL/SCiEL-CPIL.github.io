import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
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

    def build_dp(S):
        n = len(S)
        dp = [0] * (n + 1) 
        dp[0] = 1
        for x in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        return dp

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
                # Note: dp size is fixed 49, but we only use up to len(S)
                for k in range(len(S) - 1, 0, -1):
                    dp[k+1] |= (dp[k] << x)
                dp[1] |= (dp[0] << x)
                        
        return S

    def local_search(S, max_sum=24576):
        """
        Improve the set by trying to add elements or swapping.
        """
        if not S:
            return S
        
        S = S[:]
        S_set = set(S)
        dp = build_dp(S)
        
        nums = list(range(1, 513))
        candidates = [x for x in nums if x not in S_set]
        rng = random.Random(hash(tuple(sorted(S))) % 10000)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 1. Try to add a valid element
            if len(S) < 48:
                # Shuffle candidates to randomize
                rng.shuffle(candidates)
                for x in candidates:
                    if is_valid_candidate(x, S, dp, max_sum):
                        S.append(x)
                        S_set.add(x)
                        # Update DP
                        for k in range(len(S) - 1, 0, -1):
                            dp[k+1] |= (dp[k] << x)
                        dp[1] |= (dp[0] << x)
                        candidates.remove(x)
                        improved = True
                        break # Restart loop to try adding more

            # 2. If no addition, try a swap
            if not improved and len(S) > 0:
                # Try swapping a random element in S with a random candidate
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                    idx = rng.randrange(len(S))
                    u = S[idx]
                    v = rng.choice(candidates)
                    
                    # Form S_new = S \ {u} U {v}
                    # Check if v is valid with S \ {u}
                    S_minus_u = S[:idx] + S[idx+1:]
                    dp_minus_u = build_dp(S_minus_u)
                    
                    if is_valid_candidate(v, S_minus_u, dp_minus_u, max_sum):
                        # S_new is valid
                        S = S_minus_u + [v]
                        S_set = set(S)
                        dp = build_dp(S)
                        candidates.remove(v)
                        candidates.append(u)
                        improved = True
                        break

        return S

    # Generate orderings
    nums = list(range(1, 513))
    
    # 1. Ascending
    ord_asc = nums[:]
    
    # 2. Descending
    ord_desc = nums[::-1]
    
    # 3. Primes first
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5)+1):
            if x % i == 0:
                return False
        return True
    
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
    for _ in range(50):
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
            
    return best_set
