"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0  # 85s safety margin within 90s limit
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def prime_factors(n):
        """Return set of distinct prime factors of n."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Precompute prime factors for all numbers 2..256
    pf = [set() for _ in range(257)]
    for n in range(2, 257):
        pf[n] = prime_factors(n)
        
    # List of all primes up to 256
    primes = [p for p in range(2, 257) if is_prime(p)]
    
    # For each prime, count how many numbers in 2..256 are divisible by it
    # This helps prioritize "rare" primes
    prime_count = {p: 0 for p in primes}
    for n in range(2, 257):
        for p in pf[n]:
            prime_count[p] += 1
            
    # Sort primes by count (ascending) - rare primes first
    # Then by value (descending) - larger primes are generally safer?
    sorted_primes = sorted(primes, key=lambda p: (prime_count[p], -p))
    
    selected_numbers = []
    selected_set = set()
    used_primes = set() # Primes that have been "claimed" as private witnesses
    
    # We will try to assign a private prime to each number
    # Greedy: Iterate through numbers? Or through primes?
    # If we iterate through primes, we pick a number for that prime.
    # But a number might be better suited for another prime?
    # Let's iterate through numbers sorted by "difficulty" (fewer prime factors? or rarest prime factor?)
    
    # Strategy: 
    # 1. Start with empty set.
    # 2. Candidate numbers: all 2..256.
    # 3. Sort candidates by the "rarity" of their rarest prime factor.
    #    Rarity = 1 / count(prime).
    #    Higher rarity = better.
    
    def get_rarity(n):
        if not pf[n]:
            return 0
        # Find the prime factor with the highest rarity (lowest count)
        min_count = min(prime_count[p] for p in pf[n])
        return min_count
        
    candidates = list(range(2, 257))
    # Sort by min_count (ascending), then by number (descending)
    candidates.sort(key=lambda n: (get_rarity(n), -n))
    
    for n in candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Can we assign a private prime to n?
        # We need a prime p in pf[n] such that:
        # 1. p is not in used_primes
        # 2. p does not divide any already selected number
        
        valid_prime = None
        for p in sorted(pf[n], key=lambda x: prime_count[x]): # Try rarest first
            if p in used_primes:
                continue
            # Check if p divides any selected number
            conflict = False
            for s in selected_numbers:
                if s % p == 0:
                    conflict = True
                    break
            if not conflict:
                valid_prime = p
                break
        
        if valid_prime is not None:
            selected_numbers.append(n)
            selected_set.add(n)
            used_primes.add(valid_prime)
            
    # Verify the result
    # Condition: for all distinct a,b,c in S, a does not divide b*c
    # Our construction ensures each a has a private prime p_a not dividing any other b in S.
    # Thus p_a does not divide b*c. So a does not divide b*c.
    # This is a sufficient condition.
    
    # Let's double check with a brute force verifier for safety
    n = len(selected_numbers)
    valid = True
    for i in range(n):
        a = selected_numbers[i]
        for j in range(n):
            if j == i: continue
            b = selected_numbers[j]
            for k in range(n):
                if k == i or k == j: continue
                c = selected_numbers[k]
                if (b * c) % a == 0:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to primes if something went wrong (should not happen)
        return primes
        
    return selected_numbers
# EVOLVE-BLOCK-END
