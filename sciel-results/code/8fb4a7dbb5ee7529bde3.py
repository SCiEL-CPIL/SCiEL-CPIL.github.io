import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()
    
    # Build SPF sieve up to a limit that allows fast factorization
    # We need primes up to sqrt(10^12) = 10^6 for trial division
    # But for segmented approach, we can use a smaller sieve
    
    # Strategy:
    # 1. Use a segmented sieve to compute phi values for large windows
    # 2. For each window, find the longest run of distinct phi values
    # 3. Use the fact that phi(n) collisions are relatively rare for consecutive n
    #    especially when n is not highly composite
    
    # First, let's try a direct approach with a good phi function
    # that uses precomputed primes for trial division
    
    # Generate primes up to 10^6 using sieve
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def phi(x):
        """Compute Euler's totient function using precomputed primes."""
        if x <= 1:
            return x
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    # For scanning, we want to find long runs of distinct phi values
    # Let's scan in chunks and track the best run
    
    best_k = 1
    best_s = 1
    
    # Scan from 1 upward, but we need to be strategic
    # The seed found a run in range(1, 20001)
    # Let's first do a quick scan of a moderate range to establish a baseline
    
    # Quick scan: compute phi for range and find best run
    def scan_range(start, end):
        """Find the longest run of distinct phi values in [start, end]."""
        if end > 10**12:
            end = 10**12
        if start < 1:
            start = 1
        if start >= end:
            return (1, start)
        
        n = end - start + 1
        if n > 500000:  # cap the scan size
            end = start + 500000
            n = 500000
        
        # Compute phi values
        vals = [0] * (n + 1)  # 1-indexed
        for i in range(n):
            if elapsed() > deadline - 5:
                break
            vals[i + 1] = phi(start + i)
        
        # Find longest run of distinct values
        local_best_k = 1
        local_best_s = start
        
        seen = set()
        k = 0
        for i in range(1, n + 1):
            if vals[i] in seen:
                # Reset
                if k > local_best_k:
                    local_best_k = k
                    local_best_s = start + i - 1 - k
                seen = set()
                k = 0
            seen.add(vals[i])
            k += 1
            if k > local_best_k:
                local_best_k = k
                local_best_s = start + i - k
        
        return (local_best_k, local_best_s)
    
    # Phase 1: Scan a moderate range to establish baseline
    k1, s1 = scan_range(1, 200000)
    best_k, best_s = k1, s1
    
    if elapsed() > deadline - 10:
        return {'start': best_s, 'length': best_k}
    
    # Phase 2: Try larger ranges
    # The key insight: phi(n) for consecutive n are likely distinct unless
    # there's some structural reason (like n and n+1 sharing properties)
    # Let's try scanning in larger chunks
    
    # Try scanning from different starting points
    # Focus on ranges where numbers are "generic" (not highly composite)
    
    # Let's try a few different ranges
    ranges_to_try = [
        (200001, 600000),
        (600001, 1200000),
        (120001, 2000000),
        (2000001, 3000000),
    ]
    
    for start, end in ranges_to_try:
        if elapsed() > deadline - 10:
            break
        k, s = scan_range(start, end)
        if k > best_k:
            best_k, best_s = k, s
    
    # Phase 3: Try to extend the best run found
    # If we found a run of length k at position s, try to extend it
    # by checking s-1, s-2, ... and s+k, s+k+1, ...
    
    if elapsed() > deadline - 5:
        return {'start': best_s, 'length': best_k}
    
    # Try to extend the best run
    def try_extend(s, k):
        """Try to extend the run starting at s with length k."""
        # First, compute phi values in the current run
        run_vals = [phi(s + i) for i in range(k)]
        seen = set(run_vals)
        
        # Try extending to the left
        new_s = s
        for offset in range(1, 50):
            if s - offset < 1:
                break
            v = phi(s - offset)
            if v in seen:
                break
            seen.add(v)
            new_s = s - offset
        
        # Try extending to the right
        new_k = new_s + k - s  # this is k + (s - new_s)
        # Actually let's recompute
        # The run is now from new_s to s + k - 1
        # Length is (s + k - 1) - new_s + 1 = k + s - new_s
        
        # Try extending further to the right
        for offset in range(1, 50):
            idx = s + k - 1 + offset
            if idx > 10**12:
                break
            v = phi(idx)
            if v in seen:
                break
            seen.add(v)
            new_k = new_s and (idx - new_s + 1) or new_k
        
        # Recompute properly
        # The run is from new_s to s + k - 1 + (extensions to right)
        # Let's just recompute the full range
        
        # Actually, let's be more careful
        left_ext = s - new_s
        right_ext = 0
        for offset in range(1, 50):
            idx = s + k - 1 + offset
            if idx > 10**12:
                break
            v = phi(idx)
            if v in seen:
                break
            seen.add(v)
            right_ext = offset
        
        total_k = k + left_ext + right_ext
        total_s = new_s
        
        if total_k <= 200 and total_s + total_k - 1 <= 10**12:
            return (total_k, total_s)
        else:
            # Truncate if needed
            if total_s + total_k - 1 > 10**12:
                total_k = 10**12 - total_s + 1
            if total_k > 200:
                total_k = 200
            return (total_k, total_s)
    
    k_ext, s_ext = try_extend(best_s, best_k)
    if k_ext > best_k:
        best_k, best_s = k_ext, s_ext
    
    if elapsed() > deadline - 2:
        return {'start': best_s, 'length': best_k}
    
    # Phase 4: Random sampling to find potentially better runs
    # Sample random starting points and do a quick scan
    
    random.seed(42)
    for _ in range(50):
        if elapsed() > deadline - 2:
            break
        # Choose a random starting point
        s_rand = random.randint(1, min(10**7, 10**12 - 200))
        k_rand, s_rand_final = scan_range(s_rand, s_rand + 10000)
        if k_rand > best_k:
            best_k, best_s = k_rand, s_rand_final
    
    return {'start': best_s, 'length': best_k}
