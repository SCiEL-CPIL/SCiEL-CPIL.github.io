import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # Build sigma via linear sieve (O(n))
    # sigma[n] = sum of divisors of n
    # Use the multiplicative property: if n = p^k * m, gcd(p,m)=1, then sigma(n) = sigma(p^k)*sigma(m)
    # sigma(p^k) = 1 + p + p^2 + ... + p^k = (p^(k+1)-1)/(p-1)
    
    n_max = MAX_N
    sigma = [0] * (n_max + 1)
    sigma[1] = 1
    primes = []
    is_composite = [False] * (n_max + 1)
    # For linear sieve, we need to track the smallest prime factor power
    # Let's use a different approach: for each n, track p^k where p is the smallest prime factor
    # and the remaining part
    
    # Actually, let's use the standard linear sieve approach for sigma:
    # sigma is multiplicative. In a linear sieve:
    # For each i, for each prime p:
    #   if i*p > n_max: break
    #   is_composite[i*p] = True
    #   if i % p == 0:
    #     # p divides i, so p^k || i for some k >= 1
    #     # We need to track p^k for i
    #     break
    #   else:
    #     # p does not divide i, so sigma(i*p) = sigma(i) * sigma(p) = sigma(i) * (1+p)
    #     sigma[i*p] = sigma[i] * (1 + p)
    #     primes.append(p)
    
    # But we need to handle the case where p divides i. Let's use arrays:
    # lp[i] = largest prime power of smallest prime factor of i
    # sp[i] = sigma of lp[i]
    
    lp = [0] * (n_max + 1)  # lp[i] = p^k where p is smallest prime factor of i, p^k || i
    sp = [0] * (n_max + 1)  # sp[i] = sigma(lp[i])
    primes = []
    sigma[1] = 1
    sp[1] = 1
    lp[1] = 1
    
    for i in range(2, n_max + 1):
        if not is_composite[i]:
            primes.append(i)
            lp[i] = i
            sp[i] = 1 + i
            sigma[i] = 1 + i
        for p in primes:
            v = i * p
            if v > n_max:
                break
            is_composite[v] = True
            if i % p == 0:
                # p divides i, so lp[i] = p^k for some k
                lp[v] = lp[i] * p
                sp[v] = sp[i] + lp[v]
                sigma[v] = sigma[i] // sp[i] * sp[v]
                break
            else:
                lp[v] = p
                sp[v] = 1 + p
                sigma[v] = sigma[i] * (1 + p)
    
    # Now search for a, b with sigma(a) = sigma(b) minimizing |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # Let d = 2a - 3b. We want |d| small and b large.
    # a = (3b + d) / 2, so 3b + d must be even, meaning b and d have the same parity.
    
    best_error = float('inf')
    best_a = None
    best_b = None
    
    # For each b from MAX_N down to 1:
    #   a should be close to 3b/2
    #   Check a in a window around 3b/2
    #   But we need 2a - 3b = d to be small
    #   a = (3b + d)/2, so for each small d, compute a and check bounds and sigma
    
    # For b, d must have same parity as b (since 3b+d even means b+d even, so d same parity as b)
    # |d| = 1, 2, 3, ...
    # a = (3b + d) / 2 must be integer in [1, MAX_N]
    
    # Strategy: iterate b from MAX_N down, for each b try small |d| values
    # For b, d must satisfy: d ≡ b (mod 2) [since 3b+d even]
    #   and 1 <= (3b+d)/2 <= MAX_N
    #   i.e., 2 - 3b <= d <= 2*MAX_N - 3b
    
    # For large b, d range is roughly [-3b, 2*MAX_N - 3b]
    # We want small |d|, so try d = ±1, ±2, ... in order of |d|
    
    # But iterating all b is 10M iterations, and for each we try several d values.
    # Let's try d in {-5, -4, ..., 4, 5} for each b.
    
    # Actually, let's be smarter. For each small d, find b such that a = (3b+d)/2 is integer.
    # 3b + d even => b and d same parity.
    # So for fixed d, b must have parity = d % 2.
    # a = (3b+d)/2, and we need sigma(a) = sigma(b).
    
    # Let's iterate d from -20 to 20 (skip d=0 since that gives a/b = 3/2 exactly, error=0, invalid)
    # For each d, iterate b with correct parity, compute a, check bounds and sigma.
    
    # But that's still potentially many iterations. Let's limit the search.
    
    # Alternative: iterate b from MAX_N down, and for each b, check a few candidate a values
    # near 3b/2.
    
    # For b, a ≈ 3b/2. Let a = 3b//2 + offset, where offset is small.
    # 2a - 3b = 2*(3b//2 + offset) - 3b = 2*(3b//2) - 3b + 2*offset
    # If b is even: 3b//2 = 3b/2, so 2a - 3b = 3b - 3b + 2*offset = 2*offset
    # If b is odd: 3b//2 = (3b-1)/2, so 2a - 3b = (3b-1) - 3b + 2*offset = -1 + 2*offset
    # So |2a - 3b| is either 2*|offset| (b even) or |2*offset - 1| (b odd)
    
    # To get |2a-3b| = 1: b odd, offset = 0 or 1
    # To get |2a-3b| = 2: b even, offset = ±1; or b odd, offset = ±1
    # etc.
    
    # Let's iterate b from MAX_N down to 1, and for each b, try offsets -3 to 3
    # That's 7 candidates per b, times 10M b's = 70M checks. Might be too slow in Python.
    
    # Let's limit to a subset. Try b from MAX_N down, but only check every few b's,
    # or limit the number of b's we check.
    
    # Actually, let's try a different approach: iterate over sigma values.
    # Group numbers by sigma value. For each group with at least 2 numbers,
    # find pairs (a,b) in the group that minimize |a/b - 3/2|.
    
    # But building these groups for 10M numbers might use too much memory.
    
    # Let's try the direct approach but with time limits.
    
    # Iterate b from MAX_N down to 1
    # For each b, try a in a small window around 3b/2
    # Check sigma equality
    
    # To make this faster, let's precompute a lookup: for each sigma value, 
    # store the list of numbers with that sigma. But that's memory heavy.
    
    # Alternative: use a dictionary mapping sigma value to list of numbers.
    # But 10M entries might be too much.
    
    # Let's just do the direct search with time limits.
    
    # For efficiency, let's only check b values where 3b/2 is close to an integer
    # that's within bounds.
    
    # Iterate b from MAX_N down to 1
    # For b, a_target = 3*b/2
    # Check a in [a_target - 5, a_target + 5], rounded to integers
    # For each a, check 1 <= a <= MAX_N and sigma[a] == sigma[b]
    
    # But 10M * 11 = 110M checks. Too slow in pure Python.
    
    # Let's limit to a fraction of b values, or use a smarter search.
    
    # Let's try: for small |d| values (d = 2a - 3b), iterate over b.
    # For fixed d, a = (3b + d)/2.
    # We need b and d to have the same parity, and a to be an integer in [1, MAX_N].
    
    # For d in range(-10, 11):
    #   if d == 0: continue
    #   parity = d % 2
    #   b_min = max(1, (2 - d + 1) // 3)  # from 3b + d >= 2 => b >= (2-d)/3
    #   b_max = min(MAX_N, (2*MAX_N - d - 1) // 3)  # from 3b + d <= 2*MAX_N => b <= (2*MAX_N-d)/3
    #   for b in range(b_max, b_min-1, -1):
    #     if b % 2 != parity: continue
    #     a = (3*b + d) // 2
    #     if a < 1 or a > MAX_N: continue
    #     if sigma[a] == sigma[b]:
    #       error = abs(d) / (2*b)
    #       if error < best_error:
    #         best_error = error
    #         best_a = a
    #         best_b = b
    
    # This is still potentially many iterations. Let's add time checks.
    
    # Let's try a limited version: for each d, only check b in a limited range.
    # Or, iterate b from MAX_N down, and for each b, check a few a values.
    
    # Let's go with: iterate b from MAX_N down to 1, but check time every 10000 iterations.
    # For each b, check a in a window of size 11 around 3b/2.
    
    # Actually, let's try the d-based approach with a limit on b range.
    
    # For each d in a small range, iterate b from MAX_N down, but stop after checking
    # a certain number of b values or when time runs out.
    
    d_range = range(-10, 11)
    
    for d in d_range:
        if d == 0:
            continue
        if time.time() - start_time > DEADLINE:
            break
        
        parity = d % 2
        # b must satisfy: b % 2 == parity
        # a = (3b + d) / 2, 1 <= a <= MAX_N
        # 2 <= 3b + d <= 2*MAX_N
        # (2 - d) / 3 <= b <= (2*MAX_N - d) / 3
        
        b_min = max(1, math.ceil((2 - d) / 3))
        b_max = min(MAX_N, math.floor((2 * MAX_N - d) / 3))
        
        # Adjust b_max to have correct parity
        if b_max % 2 != parity:
            b_max -= 1
        # Adjust b_min to have correct parity
        if b_min % 2 != parity:
            b_min += 1
        
        if b_min > b_max:
            continue
        
        # Iterate b from b_max down to b_min
        # But this could be up to 10M iterations per d, which is too slow.
        # Let's limit to a subset.
        
        # Only check b values where b is large (to get small error).
        # Error = |d| / (2b), so for fixed d, larger b gives smaller error.
        # Let's check b from b_max down, but stop after checking a certain number
        # or when time runs out.
        
        # Check every 10th b value to reduce work? No, let's check all but with time limit.
        
        # Let's check b from b_max down to b_min, but break if time runs out.
        # Also, once we find a good solution, we can be more selective.
        
        checked = 0
        for b in range(b_max, b_min - 1, -2):  # step 2 to maintain parity
            if time.time() - start_time > DEADLINE:
                break
            
            a = (3 * b + d) // 2
            if a < 1 or a > MAX_N:
                continue
            
            if sigma[a] == sigma[b]:
                error = abs(d) / (2 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
                    # If we found a very good solution, we can tighten our search
                    if error < 1e-6:
                        # Don't break, keep looking for even better
                        pass
        
            checked += 1
            if checked % 100000 == 0:
                if time.time() - start_time > DEADLINE:
                    break
    
    if best_a is None:
        # Fallback: try a few known pairs
        # Let's check small numbers
        for b in range(1, 1000):
            for a in range(1, 1000):
                if sigma[a] == sigma[b] and a != b:
                    error = abs(a/b - 3/2)
                    if error < best_error and error > 0:
                        best_error = error
                        best_a = a
                        best_b = b
    
    if best_a is None:
        return [11, 6]
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
