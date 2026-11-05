import time
import math

N = 5000
DEADLINE = 320.0  # Safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    start_time = time.time()
    
    # Start with the full set of integers in [0, 5000]
    S = set(range(N + 1))
    current_size = len(S)
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    # We will iteratively repair the set
    # The condition is: for every prime p <= |S|, not all residues mod p are covered.
    # We check primes in increasing order. Smaller primes are more likely to be fully covered
    # and removing from them has a larger impact on reducing the "density" of the set.
    
    while current_size > 0:
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
            
        # Get primes p such that p <= current_size
        # Since all_primes is sorted, we can find the cutoff
        # We only care about primes <= current_size
        relevant_primes = [p for p in all_primes if p <= current_size]
        
        violated = False
        for p in relevant_primes:
            # Count residues for this prime p
            counts = {}
            for x in S:
                r = x % p
                counts[r] = counts.get(r, 0) + 1
            
            # Check if all p residues are covered
            if len(counts) == p:
                # Find the residue with the minimum count
                min_residue = min(counts, key=counts.get)
                min_count = counts[min_residue]
                
                # Remove all elements x in S such that x % p == min_residue
                # Using list comprehension for speed
                S = {x for x in S if x % p != min_residue}
                current_size = len(S)
                violated = True
                break # Restart the loop to re-evaluate primes with new size
        
        if not violated:
            break
    
    return sorted(S)
