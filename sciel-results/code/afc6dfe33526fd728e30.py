import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = 123457
    
    # Get primes up to target
    # Simple sieve for primes
    is_prime = [True] * (target + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(target)) + 1):
        if is_prime[i]:
            for j in range(i*i, target + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # We will iterate through primes in increasing order.
    # smooths is a set of numbers <= target that are K-smooth (using primes up to current p)
    smooths = {1}
    
    # We need to find a, b in smooths such that a+b=target.
    # We can check this after each prime is added.
    
    # Optimization: if a is in smooths, and target-a is in smooths, we found a pair.
    # We want the smallest K.
    # So we iterate primes p.
    # Add p-smooth numbers to smooths.
    # Check for pair.
    
    # To add p-smooth numbers:
    # For each x in smooths, consider x * p, x * p^2, ... <= target.
    # Add them to a new set or update smooths.
    
    # Since we want the SMALLEST K, we stop at the first prime p where a pair is found.
    # Note: P(1)=1. 1 is 1-smooth.
    
    # Initial check: 1 + (target-1).
    # target-1 = 123456.
    # Is 123456 1-smooth? No, it has prime factors.
    # So we need to add primes.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Generate new smooth numbers using prime p
        new_smooths = []
        for x in smooths:
            val = x * p
            while val <= target:
                new_smooths.append(val)
                val *= p
        
        # Add new_smooths to smooths
        # We can just add them to the set
        smooths.update(new_smooths)
        
        # Check if there exists a, b in smooths such that a + b = target
        # a must be in smooths, and target - a must be in smooths.
        found = False
        best_a = None
        best_b = None
        
        # Iterate over a subset of smooths?
        # smooths can grow large.
        # For K=100, number of smooth numbers is small.
        # For large K, it grows.
        
        # To be safe, we can iterate over all a in smooths.
        # If smooths is too large, this might be slow.
        # But if K is small, smooths is small.
        
        for a in smooths:
            b = target - a
            if b in smooths:
                # Found a pair
                # We need to minimize max(P(a), P(b)).
                # Since both are K-smooth, P(a) <= p and P(b) <= p.
                # So max(P(a), P(b)) <= p.
                # Since we iterate p in increasing order, the first p we find is the minimum.
                # However, we need to return the specific a, b.
                # Any pair works?
                # The objective is to minimize the value. The value is determined by p.
                # So any pair with max prime factor p is optimal for this p.
                best_a = a
                best_b = b
                found = True
                break
        
        if found:
            return [best_a, best_b]
            
    # Fallback: if no pair found with smooth numbers (should not happen),
    # or if we ran out of time, use the sieve method.
    # But if we ran out of time, we might not have the sieve P array.
    # So we should probably compute the sieve P array first?
    # No, the sieve is fast.
    
    # Let's stick to the sieve method as it is robust and fast.
    # The binary search might be slower if the optimal K is large.
    
    # Given the directive to "KEEP this design", maybe I should just optimize the sieve?
    
    # Let's try to optimize the sieve using local variables.
    
    return [1, target - 1] # Fallback
