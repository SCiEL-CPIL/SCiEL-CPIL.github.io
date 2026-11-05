import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    TARGET = 123457
    
    def generate_smooth_numbers(primes, max_val):
        """Generate all numbers whose prime factors are in primes, up to max_val."""
        smooth = {1}
        changed = True
        while changed:
            changed = False
            for p in primes:
                for s in list(smooth):
                    new_val = s * p
                    if new_val <= max_val and new_val not in smooth:
                        smooth.add(new_val)
                        changed = True
        return smooth
    
    def is_smooth(n, primes):
        """Check if n is smooth with respect to primes."""
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1
    
    def find_pair_for_k(k, primes, max_val):
        """Find two k-smooth numbers that sum to max_val."""
        smooth = generate_smooth_numbers(primes[:k], max_val)
        for a in smooth:
            b = max_val - a
            if b >= 1 and b in smooth:
                return (a, b)
        return None
    
    # Get all primes up to TARGET
    def sieve_primes(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve_primes(TARGET)
    
    # Binary search on K (number of smallest primes to use)
    # Lower bound: 1 (just prime 2)
    # Upper bound: len(primes)
    lo, hi = 1, len(primes)
    best_pair = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        pair = find_pair_for_k(mid, primes, TARGET)
        if pair is not None:
            best_pair = pair
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_pair is None:
        # Fallback: scan all splits
        lp = [1] * (TARGET + 1)
        for i in range(2, TARGET + 1):
            if lp[i] == 1:
                for j in range(i, TARGET + 1, i):
                    lp[j] = i
        
        best_value = float('inf')
        for a in range(1, TARGET):
            b = TARGET - a
            cost = max(lp[a], lp[b])
            if cost < best_value:
                best_value = cost
                best_pair = (a, b)
            if time.time() > deadline:
                break
    
    return list(best_pair)
