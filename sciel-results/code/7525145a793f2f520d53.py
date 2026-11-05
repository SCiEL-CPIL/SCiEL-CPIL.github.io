import itertools
from math import isqrt

def _is_smooth(n, primes):
    """Check if n is composed only of primes in the set primes."""
    if n <= 1:
        return True
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            while n % p == 0:
                n //= p
    return n == 1

def _find_k_set(K, allowed_primes, limit_nodes=200000):
    """Find 12 distinct k in [1, K] such that all pairwise sums are smooth over allowed_primes."""
    candidates = list(range(1, K + 1))
    solution = []
    nodes = 0
    
    def backtrack(start_idx, current_list):
        nonlocal nodes
        nodes += 1
        if nodes > limit_nodes:
            return False
        
        if len(current_list) == 12:
            return True
            
        for i in range(start_idx, len(candidates)):
            candidate = candidates[i]
            valid = True
            for existing in current_list:
                s = candidate + existing
                if not _is_smooth(s, allowed_primes):
                    valid = False
                    break
            if valid:
                current_list.append(candidate)
                if backtrack(i + 1, current_list):
                    return True
                current_list.pop()
        return False
    
    if backtrack(0, solution):
        return solution
    return None

def construct():
    # We want to find 12 distinct positive integers <= 10000
    # such that the number of distinct prime factors of all pairwise sums is minimized.
    
    # Strategy: a_i = 2^m * k_i
    # Sums = 2^m * (k_i + k_j)
    # Primes in sums = {2} U primes(k_i + k_j)
    # We want to minimize |{2} U primes(k_i + k_j)|.
    # This is equivalent to finding k_i such that k_i + k_j are S'-smooth
    # for a small set S'.
    
    # Try different m (powers of 2)
    # We prioritize larger K (smaller m) to allow smaller S'
    # but limit search depth.
    
    base_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    # Generate prime sets in order of size
    prime_sets = []
    for size in range(1, len(base_primes) + 1):
        for combo in itertools.combinations(base_primes, size):
            prime_sets.append(list(combo))
            
    # Try m from 6 down to 4
    for m in range(6, 3, -1):
        M = 2 ** m
        K = 10000 // M
        if K < 12:
            continue
            
        for S_prime in prime_sets:
            allowed_primes = S_prime + [2]
            k_set = _find_k_set(K, allowed_primes)
            if k_set:
                result = [M * k for k in k_set]
                if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                    return result
    
    # Fallback: simple range
    return list(range(1, 13))
