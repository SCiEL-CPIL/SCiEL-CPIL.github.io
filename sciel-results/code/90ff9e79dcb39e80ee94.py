import time
import math

N = 5000
DEADLINE = 290.0

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def construct():
    start_time = time.time()
    
    # in_set[i] is True if i is currently in the candidate set
    in_set = [True] * (N + 1)
    count = N + 1
    
    # We iterate through primes.
    # After processing prime p, the set contains numbers not divisible by any prime <= p.
    # This corresponds to avoiding residue 0 for all primes <= p.
    # The size is 'count'.
    # The admissibility condition for this set is:
    # For all primes q <= count, the set must miss a residue mod q.
    # Since the set avoids 0 for all primes <= p, it misses 0 for all q <= p.
    # If count <= p, then for all q <= count, we have q <= p, so 0 is missed.
    # Thus, the set is admissible.
    # We want the largest 'count' such that count <= p.
    # Since 'count' decreases as p increases, the first p satisfying count <= p
    # will give the maximum count.
    
    for p in PRIMES:
        if p > N:
            break
            
        # Remove multiples of p
        for multiple in range(p, N + 1, p):
            if in_set[multiple]:
                in_set[multiple] = False
                count -= 1
                
        # Check if this set is admissible
        if count <= p:
            # This is the optimal K (smallest p satisfying condition)
            # Thus 'count' is the maximum possible size for this strategy
            result = [i for i, v in enumerate(in_set) if v]
            return result
            
    # Fallback: return empty set or full set if no prime triggered the condition
    # This should not happen for N=5000
    return [i for i, v in enumerate(in_set) if v]

