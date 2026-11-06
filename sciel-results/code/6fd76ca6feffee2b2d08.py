import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin
    
    # Generate primes up to 10^6 using sieve
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def phi(x):
        """Compute Euler's totient function using trial division by primes."""
        if x == 1:
            return 1
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
    
    def check_run(s, k):
        """Check if phi(s), phi(s+1), ..., phi(s+k-1) are all distinct."""
        seen = set()
        for i in range(k):
            v = phi(s + i)
            if v in seen:
                return False, i
            seen.add(v)
        return True, k
    
    def find_best_in_range(s_start, s_end, max_k=200):
        """Find the best run in a range."""
        best_k = 1
        best_s = s_start
        for s in range(s_start, s_end + 1):
            if time.time() - start_time > DEADLINE:
                break
            # Try to extend from current best
            k = 1
            seen = {phi(s)}
            while s + k <= s_end and s + k - 1 <= 10**12 and k < max_k:
                v = phi(s + k)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
        return best_k, best_s
    
    # Strategy: denser search in strategic regions
    best_k = 1
    best_s = 1
    
    # First, do a quick search in small range to establish baseline
    k, s = find_best_in_range(1, 20000, 200)
    if k > best_k:
        best_k, best_s = k, s
    
    # Search in various strategic locations
    # We focus on 5 regions with higher density (smaller step)
    search_regions = [
        (100000, 150000),       # 10^5 range
        (10000000, 10050000),   # 10^7 range
        (1000000000, 1000050000), # 10^9 range
        (100000000000, 100000050000), # 10^11 range
        (10**12 - 50000, 10**12), # Near 10^12
    ]
    
    for lo, hi in search_regions:
        if time.time() - start_time > DEADLINE:
            break
        if lo > 10**12:
            continue
        hi = min(hi, 10**12 - 199)
        if lo >= hi:
            continue
        
        # Denser sampling: step 50
        step = 50
        for s in range(lo, hi + 1, step):
            if time.time() - start_time > DEADLINE:
                break
            
            # Check run starting at s
            k = 1
            seen = {phi(s)}
            while s + k - 1 <= 10**12 and k < 200:
                v = phi(s + k)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = s
                
                # Local refinement: check nearby starts
                # The best run might start slightly before or after s
                for ds in range(-20, 21):
                    s2 = s + ds
                    if s2 < 1 or s2 + 199 > 10**12:
                        continue
                    k2 = 1
                    seen2 = {phi(s2)}
                    while s2 + k2 - 1 <= 10**12 and k2 < 200:
                        v2 = phi(s2 + k2)
                        if v2 in seen2:
                            break
                        seen2.add(v2)
                        k2 += 1
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2
    
    # Final verification
    if best_k > 1:
        ok, _ = check_run(best_s, best_k)
        if not ok:
            # Fall back to safe answer
            best_k = 1
            best_s = 1
    
    return {'start': best_s, 'length': best_k}
