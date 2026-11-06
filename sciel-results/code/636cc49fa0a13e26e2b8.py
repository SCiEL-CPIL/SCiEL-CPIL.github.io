"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
from math import isqrt

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

PRIMES = _primes_up_to(1000)

def construct():
    """
    Construct a set of prime residue classes covering 1..L with sum 1/p <= 1.
    Uses Greedy Set Cover with NumPy acceleration.
    """
    N = 15000  # Search window
    covered = np.zeros(N + 1, dtype=bool)
    remaining_budget = Fraction(1, 1)
    used_primes = set()
    classes = []
    
    # Precompute indices for each prime and residue as NumPy arrays
    # prime_indices[p] is a list of arrays, prime_indices[p][r] is array of n in 1..N with n%p==r
    prime_indices = {}
    for p in PRIMES:
        prime_indices[p] = [None] * p
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start > N:
                prime_indices[p][r] = np.array([], dtype=np.int32)
            else:
                prime_indices[p][r] = np.arange(start, N + 1, p, dtype=np.int32)
                
    # Greedy Loop
    while True:
        best_gain = 0
        best_p = None
        best_r = None
        
        for p in PRIMES:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if cost > remaining_budget:
                continue
                
            max_gain_for_p = 0
            best_r_for_p = -1
            
            for r in range(p):
                idx = prime_indices[p][r]
                if len(idx) == 0:
                    continue
                
                # Count uncovered
                gain = np.sum(~covered[idx])
                
                if gain > max_gain_for_p:
                    max_gain_for_p = gain
                    best_r_for_p = r
                    
            if max_gain_for_p > best_gain:
                best_gain = max_gain_for_p
                best_p = p
                best_r = best_r_for_p
                
        if best_p is None or best_gain == 0:
            break
            
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        remaining_budget -= Fraction(1, best_p)
        
        # Update covered
        idx = prime_indices[best_p][best_r]
        covered[idx] = True
        
    # Determine actual L
    # Scan 1..100000
    L = 0
    for n in range(1, 100001):
        is_cov = False
        for p, r in classes:
            if n % p == r:
                is_cov = True
                break
        if is_cov:
            L = n
        else:
            break
            
    return classes

# EVOLVE-BLOCK-END
