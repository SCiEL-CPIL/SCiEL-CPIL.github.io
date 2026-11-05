import time
import math

N = 5000
DEADLINE = 315.0  # Safety margin before 330s kill

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
    
    # Iteratively repair: for each prime p <= current_size, ensure not all residues are covered
    while current_size > 0:
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
        
        # Find relevant primes: p <= current_size
        # Since all_primes is sorted, find the largest index where prime <= current_size
        # Use binary search or just iterate until we find p > current_size
        relevant_primes = []
        for p in all_primes:
            if p <= current_size:
                relevant_primes.append(p)
            else:
                break
        
        violated = False
        for p in relevant_primes:
            # Count residues mod p using an array for speed
            counts = [0] * p
            for x in S:
                counts[x % p] += 1
            
            # Check if any residue class is empty
            if 0 in counts:
                continue
            
            # All residues are covered. Remove the residue class with minimum count.
            min_count = min(counts)
            # Find the first residue with minimum count
            target_residue = counts.index(min_count)
            
            # Remove all elements x in S such that x % p == target_residue
            S = {x for x in S if x % p != target_residue}
            current_size = len(S)
            violated = True
            break  # Restart to re-evaluate with new size
        
        if not violated:
            break
    
    return sorted(S)
