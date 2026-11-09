import math
import random

def construct():
    # Sieve of Eratosthenes for primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # Precompute phi for a range of numbers using sieve method
    # This is much faster than trial division for each number
    def phi_range(start, length):
        """
        Compute phi(n) for n in [start, start+length-1]
        Uses sieve-like approach: for each prime p, divide all multiples in range
        """
        if length <= 0:
            return []
        
        # Initialize phi values and remaining values
        phi_vals = list(range(start, start + length))
        rem_vals = list(range(start, start + length))
        
        # For each prime, update phi for multiples
        for p in primes:
            if p * p > start + length - 1:
                # Still need to handle large primes that are factors
                # But for now, we'll handle remaining > 1 at the end
                pass
            
            # Find the first multiple of p in [start, start+length-1]
            # first = ceil(start/p) * p
            if start % p == 0:
                first = start
            else:
                first = start + (p - start % p)
            
            if first > start + length - 1:
                continue
                
            # Iterate through multiples of p in the range
            for j in range(first, start + length, p):
                idx = j - start
                if rem_vals[idx] % p == 0:
                    phi_vals[idx] = phi_vals[idx] // p * (p - 1)
                    while rem_vals[idx] % p == 0:
                        rem_vals[idx] //= p
        
        # Handle remaining factors > 1
        for i in range(length):
            if rem_vals[i] > 1:
                phi_vals[i] = phi_vals[i] // rem_vals[i] * (rem_vals[i] - 1)
        
        return phi_vals
    
    # We want to maximize k. Max possible is 200.
    # Search in high range where collisions are rare.
    best_k = 1
    best_s = 1
    
    # Try random starting points in [10^11, 10^12]
    # With fast phi_range, we can try many more attempts
    for _ in range(200):  # 200 attempts
        s = random.randint(10**11, 10**12 - 200)
        
        # Compute phi for all 200 numbers at once
        phi_values = phi_range(s, 200)
        
        # Check for distinctness
        seen = set()
        k = 0
        for val in phi_values:
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
                
    return {'start': best_s, 'length': best_k}
