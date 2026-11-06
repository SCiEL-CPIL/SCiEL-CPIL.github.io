import time
import random
from math import isqrt

def _factorize(n):
    """Return the set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Strategy: Use a base K such that K has small prime factors.
    # Let a_i = K * m_i.
    # Sums are K * (m_i + m_j).
    # Prime factors = primes(K) U primes(m_i + m_j).
    
    # Try K = 210 = 2 * 3 * 5 * 7
    K = 210
    max_m = 10000 // K
    # We need 12 distinct m_i in [1, max_m]
    # max_m = 47
    
    # Precompute prime factors for all possible sums m_i + m_j
    # Max sum is 46 + 47 = 93
    sum_factors = {}
    for s in range(2, 2 * max_m + 1):
        sum_factors[s] = _factorize(s)
        
    base_primes = _factorize(K)
    
    best_score = float('inf')
    best_set = None
    
    # Greedy search with multiple restarts
    # We will try to find a set of 12 multipliers that minimizes the union of prime factors.
    
    def greedy_search(candidates, initial_set=None):
        if initial_set is None:
            initial_set = []
        current_set = list(initial_set)
        current_primes = set(base_primes)
        
        # Update current_primes with sums from initial_set
        for i in range(len(current_set)):
            for j in range(i+1, len(current_set)):
                s = current_set[i] + current_set[j]
                current_primes.update(sum_factors[s])
                
        while len(current_set) < 12:
            best_candidate = None
            best_new_primes = None
            best_count = float('inf')
            
            for m in candidates:
                if m in current_set:
                    continue
                # Calculate new primes
                new_primes = set(current_primes)
                for x in current_set:
                    s = m + x
                    new_primes.update(sum_factors[s])
                
                count = len(new_primes)
                if count < best_count:
                    best_count = count
                    best_candidate = m
                    best_new_primes = new_primes
            if best_candidate is None:
                return None, current_primes
            current_set.append(best_candidate)
            current_primes = best_new_primes
            
        return current_set, current_primes

    # We will run multiple restarts with random shuffles of candidates
    candidates_base = list(range(1, max_m + 1))
    
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates_base)
        # Start with an empty set or a random small set?
        # Empty set is fine, greedy will pick the best first element.
        # However, the first element doesn't affect the prime union (no pairs).
        # So the first pick is arbitrary. We can just pick the first candidate.
        # But subsequent picks depend on it.
        
        set_found, primes_found = greedy_search(candidates_base)
        
        if set_found is not None:
            score = len(primes_found)
            if score < best_score:
                best_score = score
                best_set = set_found
                
        # Early stop if we found a perfect set (only base primes)
        if best_score == len(base_primes):
            break

    if best_set is None:
        # Fallback: first 12 multiples
        best_set = list(range(1, 13))
        
    # Construct the final numbers
    result = [K * m for m in best_set]
    
    # Verify distinctness and limit
    if len(set(result)) != 12 or any(x > 10000 for x in result):
        return list(range(1, 13))
        
    return result
