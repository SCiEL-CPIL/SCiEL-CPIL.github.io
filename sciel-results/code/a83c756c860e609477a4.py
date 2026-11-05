import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 10s safety margin
    
    # Generate primes up to 10^7 using segmented sieve or simple sieve
    # For phi computation up to 10^12, we need primes up to sqrt(10^12) = 10^6
    # But for better factoring, let's go up to 10^7
    limit = 10**7
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def phi(x):
        """Compute Euler's totient function using trial division with precomputed primes."""
        if x <= 1:
            return x
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result
    
    def check_run(s, k):
        """Check if phi(s), ..., phi(s+k-1) are all distinct. Returns length of valid run."""
        seen = set()
        for i in range(k):
            val = phi(s + i)
            if val in seen:
                return i
            seen.add(val)
        return k
    
    # Strategy: Search for long runs of distinct totients
    # Key insight: collisions happen when phi(a) = phi(b) for a != b
    # This is common when a and b share similar prime factor structures
    
    # First, do a quick scan of small ranges to find good starting points
    # Then verify with exact computation
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Scan small numbers to understand collision patterns
    # Compute phi for 1..100000 and find max run
    max_scan = 100000
    phi_values = [0] * (max_scan + 1)
    for i in range(1, max_scan + 1):
        phi_values[i] = phi(i)
    
    for s in range(1, max_scan - 200):
        seen = set()
        k = 0
        while s + k <= max_scan and phi_values[s + k] not in seen:
            seen.add(phi_values[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Search in larger ranges using probabilistic filtering
    # Avoid numbers where collisions are likely
    # Collision-prone patterns:
    # - Consecutive even numbers (both have factor 2)
    # - n where n-1 is prime (phi(n-1) = n-2, which may collide)
    # - Numbers with many small prime factors
    
    # Use random sampling in larger ranges
    search_ranges = []
    
    # Check around known good spots and random locations
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        # Random start in range [10^6, 10^10]
        s = random.randint(10**6, 10**10)
        
        # Quick probabilistic check: estimate collision risk
        # If s is even, phi(s) is even; if s+1 is even, phi(s+1) is even
        # More collisions likely when consecutive numbers have similar structure
        
        # Just try to verify a run of length up to 200
        k_found = check_run(s, min(200, best_k + 50))
        if k_found > best_k:
            best_k = k_found
            best_s = s
    
    # Phase 3: Targeted search near the best found so far
    if time.time() < deadline - 10:
        # Search in windows around best_s
        for offset in range(-5000, 5001, 100):
            if time.time() > deadline - 10:
                break
            s = best_s + offset
            if s < 1:
                continue
            k_found = check_run(s, min(200, best_k + 50))
            if k_found > best_k:
                best_k = k_found
                best_s = s
    
    # Phase 4: Broader random search
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Search in different magnitude ranges
            magnitude = random.choice([10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10, 10**11])
            s = random.randint(magnitude, min(magnitude * 10, 10**12 - 200))
            
            k_found = check_run(s, min(200, best_k + 50))
            if k_found > best_k:
                best_k = k_found
                best_s = s
    
    # Final verification of the best result
    if time.time() < deadline:
        # Verify best_s has at least best_k distinct totients
        verified = check_run(best_s, best_k)
        if verified < best_k:
            # Fallback: use the small scan result
            best_k = min(best_k, verified)
            # Find a valid run of length best_k
            seen = set()
            k = 0
            while best_s + k <= max_scan and phi_values[best_s + k] not in seen:
                seen.add(phi_values[best_s + k])
                k += 1
            best_k = min(best_k, k)
    
    # Ensure constraints are met
    best_k = min(best_k, 200)
    best_s = max(1, best_s)
    if best_s + best_k - 1 > 10**12:
        best_s = 10**12 - best_k + 1
    
    return {'start': best_s, 'length': best_k}
