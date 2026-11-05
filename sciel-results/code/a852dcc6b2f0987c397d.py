import math
import time
import hashlib

def construct():
    start_time = time.time()
    LIMIT = 115.0  # seconds (safety margin before 120s kill)
    MAX_S = 10**12

    # Generate primes up to 10^6 for trial division
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # Segmented phi computation for a window [start, start+W-1]
    def segmented_phi(start, W):
        """Compute phi(n) for n in [start, start+W-1] using segmented approach."""
        vals = list(range(start, start + W))
        phi = list(range(start, start + W))
        
        max_p_for_sieve = min(W, limit_p)
        
        for p in primes:
            if p > max_p_for_sieve:
                break
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            for j in range(first, start + W, p):
                idx = j - start
                if vals[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while vals[idx] % p == 0:
                        vals[idx] //= p
        
        for i in range(W):
            if vals[i] > 1:
                phi[i] -= phi[i] // vals[i]
        
        return phi

    def is_probably_safe(s, W):
        """
        Probabilistic pre-screen: check if window [s, s+W-1] is likely to have
        distinct totients by checking for high-risk collision patterns.
        
        High-risk patterns:
        1. n even, n-1 prime: phi(n) = n/2 * prod, phi(n-1) = n-2. 
           Collision if n/2 * prod = n-2, i.e., n/2 * prod - n = -2
        2. Both n and n+1 share small prime structure that could lead to similar phi values
        3. n is a power of 2: phi(2^k) = 2^(k-1), which might collide with other values
        
        We use a quick heuristic: for each pair (n, n+1) in the window,
        check if they have "dangerous" structures that increase collision probability.
        """
        # Quick check: avoid windows where many consecutive numbers are "smooth"
        # or have very few prime factors (which makes phi values more predictable)
        
        # For a window of size W, check a subset of positions for high-risk patterns
        check_positions = range(0, W, max(1, W // 20))  # Check ~20 positions
        
        for offset in check_positions:
            n = s + offset
            
            # Check if n is a power of 2 (high risk for phi collisions)
            if n > 0 and (n & (n - 1)) == 0:
                return False  # n is power of 2, risky
                
            # Check if n-1 is prime and n is even (classic collision risk)
            # phi(n) for even n: phi(n) = n * prod(1-1/p) for p|n
            # phi(n-1) = n-2 if n-1 is prime
            # These can collide when n/2 * (prod for odd factors) = n-2
            if n % 2 == 0 and n > 2:
                # Quick primality test for n-1 (only for small factors)
                m = n - 1
                if m > 1:
                    is_m_prime = True
                    for p in primes[:20]:  # Check small primes only for speed
                        if p * p > m:
                            break
                        if m % p == 0:
                            if m // p == 1:
                                is_m_prime = True
                            else:
                                is_m_prime = False
                            break
                    if is_m_prime:
                        # n-1 is prime, n is even: check if phi values could collide
                        # phi(n) = n/2 * phi(n/2) if n = 2 * (n/2)
                        # This is a heuristic risk marker
                        pass  # Don't reject immediately, just mark as higher risk
        
        return True

    def check_window(s):
        """Check if phi(s)..phi(s+WINDOW-1) are all distinct."""
        if s < 1 or s + WINDOW - 1 > MAX_S:
            return False
        phis = segmented_phi(s, WINDOW)
        seen = set()
        for v in phis:
            if v in seen:
                return False
            seen.add(v)
        return True

    def find_best_in_block(s, block_size):
        """Find the longest run of distinct phi values in [s, s+block_size-1]."""
        if s < 1 or s + block_size - 1 > MAX_S:
            return (1, s)
        
        phis = segmented_phi(s, block_size)
        
        seen_vals = {}
        l = 0
        max_len = 0
        max_start_idx = 0
        
        for r in range(len(phis)):
            v = phis[r]
            if v in seen_vals:
                prev_idx = seen_vals[v]
                if prev_idx >= l:
                    l = prev_idx + 1
            seen_vals[v] = r
            
            curr_len = r - l + 1
            if curr_len > max_len:
                max_len = curr_len
                max_start_idx = l
                if max_len >= 200:
                    break
        
        return (max_len, s + max_start_idx)

    WINDOW = 200
    best_k = 1
    best_s = 1

    # Phase 1: Use probabilistic pre-screen to find safe windows near 10^12
    # Generate candidate positions and pre-screen them
    candidates = []
    
    # Near 10^12 with various offsets
    for offset in range(0, 100000, 100):
        pos = MAX_S - WINDOW - offset
        if pos >= 1:
            candidates.append(pos)
    
    # Deterministic pseudo-random positions across the range
    for i in range(200):
        h = int(hashlib.md5(f"pos_{i}".encode()).hexdigest(), 16)
        pos = 1 + h % (MAX_S - WINDOW)
        candidates.append(pos)
    
    # Pre-screen candidates
    screened = []
    for s in candidates:
        if time.time() - start_time > LIMIT - 20:
            break
        if is_probably_safe(s, WINDOW):
            screened.append(s)
    
    # Verify screened candidates
    for s in screened:
        if time.time() - start_time > LIMIT - 10:
            break
        if check_window(s):
            return {'start': s, 'length': 200}
    
    # Phase 2: If pre-screened candidates didn't work, do broader scan
    # Use larger blocks for better coverage
    block_size = 3000
    scan_regions = [
        (MAX_S - block_size, MAX_S),
        (MAX_S - 10**7 - block_size, MAX_S - 10**7),
        (MAX_S - 10**8 - block_size, MAX_S - 10**8),
        (MAX_S - 10**9 - block_size, MAX_S - 10**9),
        (10**6, 10**6 + block_size),
        (10**8, 10**8 + block_size),
        (10**10, 10**10 + block_size),
    ]
    
    for region_start, region_end in scan_regions:
        if time.time() - start_time > LIMIT - 5:
            break
        if region_start < 1 or region_end > MAX_S:
            continue
        cur = region_end
        while cur >= region_start:
            if time.time() - start_time > LIMIT - 5:
                break
            block_s = max(region_start, cur - block_size + 1)
            if block_s < 1:
                block_s = 1
            actual_size = min(block_size, cur - block_s + 1)
            k, s = find_best_in_block(block_s, actual_size)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    return {'start': best_s, 'length': 200}
            cur = block_s - 1
    
    return {'start': best_s, 'length': best_k}
