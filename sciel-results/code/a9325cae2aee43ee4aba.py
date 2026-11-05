"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin before 150s limit
    
    # Precompute primes up to 10^6 for efficient factorization
    # For n <= 10^12, after removing all prime factors <= 10^6,
    # the remainder is either 1 or prime (since p*q > 10^12 if p,q > 10^6)
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    if limit_p >= 1:
        is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = 0
    
    # Extract primes list for trial division
    primes = [i for i in range(2, limit_p + 1) if is_prime[i]]
    
    def segmented_phi(s, k):
        """
        Compute phi(s), phi(s+1), ..., phi(s+k-1) using segmented sieve.
        Returns list of k totient values.
        """
        if k <= 0:
            return []
        
        # Initialize arrays for the segment [s, s+k-1]
        phi_vals = [0] * k
        remaining = [0] * k
        
        for i in range(k):
            phi_vals[i] = s + i
            remaining[i] = s + i
        
        # For each prime p <= sqrt(s+k-1), mark multiples in the segment
        max_n = s + k - 1
        sqrt_max = int(math.isqrt(max_n)) + 1
        
        for p in primes:
            if p > sqrt_max:
                break
            
            # Find the first multiple of p in [s, s+k-1]
            first_mult = ((s + p - 1) // p) * p
            
            for m in range(first_mult, max_n + 1, p):
                idx = m - s
                if remaining[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors > sqrt_max
        for i in range(k):
            if remaining[i] > 1:
                phi_vals[i] -= phi_vals[i] // remaining[i]
        
        return phi_vals
    
    def check_run(s, max_k=200):
        """Check phi(s), ..., phi(s+max_k-1) for pairwise distinctness. Return run length."""
        phi_values = segmented_phi(s, max_k)
        seen = set()
        k = 0
        for val in phi_values:
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    def check_run_extended(s, max_k=200, extra=50):
        """Check run starting at s, allowing up to max_k+extra to find best sub-window."""
        total = max_k + extra
        phi_values = segmented_phi(s, total)
        seen = set()
        k = 0
        for val in phi_values:
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Quick scan of small numbers
    small_limit = 5000
    small_phi = segmented_phi(1, small_limit)
    for s in range(1, small_limit - 200):
        seen, k = set(), 0
        while s + k - 1 <= small_limit and small_phi[s + k - 1] not in seen:
            seen.add(small_phi[s + k - 1])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        if best_k >= 200:
            break
    
    if time.time() > DEADLINE - 10:
        return {'start': best_s, 'length': best_k}
    
    # Phase 2: Deterministic sweep using primorial-based arithmetic progression
    # Primorial 2*3*5*7*11*13*17*19*23*29*31 = 20086291530
    # Numbers near multiples of this primorial have many small prime factors,
    # which tends to produce more distinct totient values.
    
    primorial = 20086291530  # 2*3*5*7*11*13*17*19*23*29*31
    
    # Generate candidate starting points around multiples of the primorial
    # in the range [10^6, 10^12]
    candidates = []
    
    # Multiples of primorial in range
    min_mult = max(1, (10**6 + primorial - 1) // primorial)
    max_mult = (10**12 - 200) // primorial
    
    # Sample a subset of multiples (can't check all ~50000)
    # Use a stride to sample ~500 candidates
    num_multiples = max_mult - min_mult + 1
    stride = max(1, num_multiples // 500)
    
    for mult in range(min_mult, max_mult + 1, stride):
        base = mult * primorial
        # Try several offsets around each multiple
        for offset in range(-20, 21, 4):
            s = base + offset
            if 1 <= s and s + 200 <= 10**12:
                candidates.append(s)
    
    # Also add some random large numbers for diversity
    random.seed(42)
    for _ in range(200):
        s = random.randint(10**7, 10**12 - 200)
        candidates.append(s)
    
    # Shuffle to interleave primorial-based and random candidates
    random.shuffle(candidates)
    
    # Check candidates
    checked = 0
    for s in candidates:
        if time.time() > DEADLINE - 15:
            break
        
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 200:
                break
        
        checked += 1
    
    # Phase 3: Local optimization around best start
    if best_k < 200 and time.time() < DEADLINE - 10:
        # Try neighbors of the best start with a wider window
        for offset in range(-100, 101, 2):
            if time.time() > DEADLINE - 5:
                break
            s = best_s + offset
            if s < 1 or s + 200 > 10**12:
                continue
            
            k = check_run(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    break
        
        # Also try extended windows to find better sub-runs
        if time.time() < DEADLINE - 10:
            for offset in range(-200, 201, 10):
                if time.time() > DEADLINE - 5:
                    break
                s = best_s + offset
                if s < 1 or s + 250 > 10**12:
                    continue
                
                k = check_run_extended(s, 200, 50)
                if k > best_k:
                    best_k = k
                    best_s = s
                    if best_k >= 200:
                        break
    
    # Phase 4: Final targeted search if still below 200
    if best_k < 200 and time.time() < DEADLINE - 8:
        # Focus on very large numbers where collisions are rarest
        random.seed(123)
        for _ in range(500):
            if time.time() > DEADLINE - 5:
                break
            s = random.randint(10**10, 10**12 - 200)
            k = check_run(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    break
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
