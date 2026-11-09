import math
import time
from itertools import combinations

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number (product of distinct primes)."""
    divisors = [1]
    for p in primes:
        new_divs = [d * p for d in divisors]
        divisors.extend(new_divs)
    divisors.sort()
    return divisors

def _tau_perp_squarefree(primes):
    """Count adjacent coprime pairs among divisors of a squarefree number."""
    divs = _divisors_of_squarefree(primes)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _score_primes(prime_list):
    """Score a list of primes as a squarefree number."""
    n = 1
    for p in prime_list:
        n *= p
    if n > 10**18:
        return -1, n
    return _tau_perp_squarefree(prime_list), n

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First 20 primes
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    best_score = -1
    best_n = 2
    
    # Try all subsets of size 14 and 15 from the first 16 primes
    # This is manageable: C(16,14) + C(16,15) + C(16,16) = 120 + 16 + 1 = 137
    # Plus we can try some with larger primes
    
    # Start with the primorial of first 15 primes as baseline
    base_primes = all_primes[:15]
    base_score, base_n = _score_primes(base_primes)
    if base_n <= 10**18:
        best_score = base_score
        best_n = base_n
    
    # Try swapping one prime in base with a larger prime
    for i in range(15):
        for j in range(15, 20):
            candidate = list(base_primes)
            candidate[i] = all_primes[j]
            candidate.sort()
            s, n = _score_primes(candidate)
            if n <= 10**18 and s > best_score:
                best_score = s
                best_n = n
            if time.time() > deadline:
                return best_n
    
    # Try swapping two primes
    for i in range(15):
        for j in range(i + 1, 15):
            for k in range(15, 20):
                for l in range(k + 1, 20):
                    candidate = list(base_primes)
                    candidate[i] = all_primes[k]
                    candidate[j] = all_primes[l]
                    candidate.sort()
                    s, n = _score_primes(candidate)
                    if n <= 10**18 and s > best_score:
                        best_score = s
                        best_n = n
                    if time.time() > deadline:
                        return best_n
    
    # Also try subsets of size 14 with larger primes
    for subset in combinations(all_primes[:18], 14):
        s, n = _score_primes(list(subset))
        if n <= 10**18 and s > best_score:
            best_score = s
            best_n = n
        if time.time() > deadline:
            return best_n
    
    return best_n
