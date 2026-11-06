import time
import math
from math import gcd

_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _elapsed(t0):
    return time.time() - t0

def _time_exhausted(t0, margin=2.0):
    return _elapsed(t0) > (_DEADLINE - margin)

def _score(chosen):
    """Compute sum 1/(512 - a) as float."""
    s = 0.0
    for a in chosen:
        s += 1.0 / (512.0 - a)
    return s

def _get_prime_factors(n, primes):
    """Get unique prime factors of n."""
    factors = set()
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.add(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.add(temp)
    return factors

def construct():
    t0 = time.time()
    
    # Sieve of Eratosthenes for primes up to 512
    limit = 512
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Pre-compute prime factors for all numbers 1..511
    prime_factors = [set() for _ in range(limit)]
    for i in range(1, limit):
        prime_factors[i] = _get_prime_factors(i, primes)
        
    # Greedy Initialization
    # We want to pick numbers with large 1/(512-a), i.e., large a.
    # Check if a is coprime to all chosen.
    # This is equivalent to: prime_factors[a] is disjoint from union of prime_factors[chosen].
    
    chosen = []
    chosen_prime_factors = set()
    
    for a in range(511, 0, -1):
        if not prime_factors[a]: # 1 has no prime factors, gcd(1, x) = 1 always
            chosen.append(a)
            # 1 doesn't add any prime factors to the forbidden set
        else:
            # Check disjointness
            if not prime_factors[a].intersection(chosen_prime_factors):
                chosen.append(a)
                chosen_prime_factors.update(prime_factors[a])
                
    best_score = _score(chosen)
    
    # Local Search
    # We maintain the list of chosen numbers.
    # We try to swap any element with a better one.
    
    improved = True
    while improved and not _time_exhausted(t0):
        improved = False
        chosen_set = set(chosen)
        
        for i in range(len(chosen)):
            if _time_exhausted(t0):
                break
            
            current = chosen[i]
            current_score = 1.0 / (512.0 - current)
            
            # Determine required prime factors (from all other chosen elements)
            req_factors = set()
            for j in range(len(chosen)):
                if j != i:
                    req_factors.update(prime_factors[chosen[j]])
            
            # Find the best replacement
            best_new = -1
            best_new_score = 0.0
            
            # Iterate candidates in descending order of score (descending a)
            # We can stop early if we find a valid one that is better?
            # Since we iterate descending a, the first valid one is the best valid one for this slot.
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if a == current:
                    continue
                
                # Check if a is coprime to all other chosen
                if prime_factors[a] and prime_factors[a].intersection(req_factors):
                    continue
                
                # Check if a is better
                new_score = 1.0 / (512.0 - a)
                if new_score > current_score:
                    best_new = a
                    best_new_score = new_score
                    break # Found the best candidate for this slot
            
            if best_new != -1:
                # Perform swap
                chosen[i] = best_new
                chosen_set.remove(current)
                chosen_set.add(best_new)
                best_score = best_score - current_score + best_new_score
                improved = True
                # Break out of the inner loop to restart the outer loop
                # This ensures we pick up any new conflicts or opportunities
                # created by the swap, and maintains a simple convergence logic.
                break
    
    return chosen
