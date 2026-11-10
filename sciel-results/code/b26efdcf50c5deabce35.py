import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate primes up to 10^6 (sqrt of 10^12)
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    
    def compute_phi_segment(s, k):
        """Compute phi(s), phi(s+1), ..., phi(s+k-1) using segmented sieve."""
        # phi[i] represents phi(s+i)
        phi = list(range(s, s + k))
        remaining = list(range(s, s + k))
        
        for p in primes:
            if p * p > s + k - 1:
                break
            # Find first multiple of p in [s, s+k-1]
            first = ((s + p - 1) // p) * p
            for j in range(first, s + k, p):
                idx = j - s
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors (each remaining value > 1 is prime)
        for i in range(k):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi
    
    def check_distinct(phi_vals):
        """Check if all values in phi_vals are distinct."""
        seen = set()
        for v in phi_vals:
            if v in seen:
                return False
            seen.add(v)
        return True
    
    # Search strategy: try high ranges first
    # We want s + k - 1 <= 10^12, so s <= 10^12 - 199
    max_s = 10**12 - 199
    
    best_k = 1
    best_s = 1
    
    # Try a set of candidate starting points in high ranges
    # Use spaced-out positions to maximize diversity
    candidates = []
    
    # High range: near 10^12
    base_high = 10**12 - 10**9
    # Try positions spaced by large gaps
    for offset in range(0, 50):
        s = base_high + offset * 10**7
        if s <= max_s:
            candidates.append(s)
    
    # Also try some mid-high ranges
    for base in [5 * 10**11, 3 * 10**11, 10**11]:
        for offset in range(0, 20):
            s = base + offset * 10**8
            if s <= max_s and s >= 1:
                candidates.append(s)
    
    # Sort candidates in descending order (high ranges first)
    candidates = sorted(set(candidates), reverse=True)
    
    for s in candidates:
        if time.time() > deadline:
            break
        
        if s < 1 or s > max_s:
            continue
        
        phi_vals = compute_phi_segment(s, 200)
        
        if check_distinct(phi_vals):
            best_k = 200
            best_s = s
            break
        
        # If not all 200 distinct, find the longest prefix that is distinct
        seen = set()
        k = 0
        for i, v in enumerate(phi_vals):
            if v in seen:
                break
            seen.add(v)
            k = i + 1
        
        if k > best_k:
            best_k = k
            best_s = s
    
    # If we haven't found 200, do a more thorough search
    if best_k < 200:
        # Try more random-ish positions in high range
        import random
        random.seed(42)
        for _ in range(200):
            if time.time() > deadline:
                break
            
            s = random.randint(10**11, max_s)
            phi_vals = compute_phi_segment(s, 200)
            
            if check_distinct(phi_vals):
                best_k = 200
                best_s = s
                break
            
            seen = set()
            k = 0
            for i, v in enumerate(phi_vals):
                if v in seen:
                    break
                seen.add(v)
                k = i + 1
            
            if k > best_k:
                best_k = k
                best_s = s
    
    return {'start': best_s, 'length': best_k}
