import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Sieve primes up to 10^6 for fast factorization
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = 0
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def phi(x):
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
    
    # Search in high ranges where totient collisions are rare
    # We want to find a run of length 200 (the maximum allowed)
    # Start searching from high values
    
    best_k = 1
    best_s = 1
    
    # Try several high starting points
    # The idea: for large n, phi(n) values in a short interval are very likely distinct
    # We'll check windows of size 200 at various high starting points
    
    # Strategy: check windows starting at s where s is around 10^11 to 10^12
    # We'll sample several positions
    
    max_n = 10**12
    window_size = 200
    
    # Try a few strategic starting points
    # We'll try points that are "random-looking" but deterministic
    # to avoid any bias
    
    candidates = []
    
    # Spread out candidates across the range [10^10, 10^12 - 200]
    # Use a deterministic pseudo-random selection
    
    # First, try the very top of the range
    candidates.append(max_n - window_size)
    
    # Then add some spread out points
    base = 10**10
    step = (max_n - window_size - base) // 10
    for i in range(10):
        s = base + i * step
        if s + window_size <= max_n:
            candidates.append(s)
    
    # Also try some points in the middle
    mid = (10**10 + max_n) // 2
    for offset in [-1000000, -500000, 0, 500000, 1000000]:
        s = mid + offset
        if s >= 1 and s + window_size <= max_n:
            candidates.append(s)
    
    # Sort candidates and deduplicate
    candidates = sorted(set(candidates))
    
    for s in candidates:
        if time.time() > deadline:
            break
        if s < 1 or s + window_size > max_n:
            continue
        
        # Check if phi(s), phi(s+1), ..., phi(s+199) are all distinct
        seen = set()
        valid = True
        for i in range(window_size):
            val = phi(s + i)
            if val in seen:
                valid = False
                break
            seen.add(val)
        
        if valid:
            best_k = window_size
            best_s = s
            break  # Found a full window of 200, which is the maximum
    
    # If we didn't find a full 200, try to find the longest run
    # by checking shorter windows at various positions
    if best_k < window_size:
        # Try to extend from candidate positions with shorter windows
        for s in candidates:
            if time.time() > deadline:
                break
            if s < 1 or s + window_size > max_n:
                continue
            
            seen = set()
            k = 0
            for i in range(window_size):
                val = phi(s + i)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = s
    
    # If still not great, do a more thorough search in a smaller range
    if best_k < 100 and time.time() < deadline - 10:
        # Search in a moderate range with step
        search_start = 10**9
        search_end = min(10**10, max_n - window_size)
        step = 10000
        
        s = search_start
        while s + window_size <= search_end:
            if time.time() > deadline:
                break
            
            seen = set()
            k = 0
            for i in range(window_size):
                val = phi(s + i)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = s
            
            s += step
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
