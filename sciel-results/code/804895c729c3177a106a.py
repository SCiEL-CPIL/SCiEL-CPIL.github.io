import time
import sys

N = 5000
START_TIME = time.time()
TIME_LIMIT = 290.0  # Leave 10s buffer for 300s limit

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    primes = _primes(N)
    active = [True] * (N + 1)
    count = N + 1
    
    # We will iterate until no more removals are needed or time runs out
    while True:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Get current length
        current_len = count
        
        # Find the smallest prime p <= current_len that violates the condition
        # i.e., all residue classes mod p are present in S
        violation_found = False
        for p in primes:
            if p > current_len:
                break
            
            # Count residues
            counts = [0] * p
            # Iterate through active elements
            # Optimization: Only iterate if necessary? No, we need counts.
            # Using list comprehension or loop. Loop is fine for N=5000.
            for i in range(N + 1):
                if active[i]:
                    counts[i % p] += 1
            
            if min(counts) > 0:
                # All classes are covered. We must remove one.
                # Remove the class with the minimum count.
                min_count = min(counts)
                # Find the first class with this count
                r = counts.index(min_count)
                
                # Remove elements in this class
                # active[i] = False for i in r, r+p, r+2p...
                removed = 0
                for i in range(r, N + 1, p):
                    if active[i]:
                        active[i] = False
                        removed += 1
                
                count -= removed
                violation_found = True
                break # Restart the prime loop with the new set
        
        if not violation_found:
            # No violations found, we are done
            break
    
    # Construct the final list
    return [i for i in range(N + 1) if active[i]]
