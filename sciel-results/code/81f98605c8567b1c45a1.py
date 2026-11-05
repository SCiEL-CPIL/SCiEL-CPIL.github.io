import time
from math import gcd

def construct():
    deadline = time.time() + 145  # Safety margin
    
    # Precompute prime factors for 1..511
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            for m in range(p, 512, p):
                prime_factors[m].add(p)
    
    # List of all candidates 1..511
    candidates = list(range(1, 512))
    
    # Sort by value descending (largest a first)
    def value(a):
        return 1.0 / (512 - a)
        
    candidates.sort(key=value, reverse=True)
    
    # Precompute values
    vals = [value(a) for a in candidates]
    factors = [prime_factors[a] for a in candidates]
    
    best_set = []
    best_val = 0.0
    best_used_primes = set()
    
    # Branch and Bound
    # idx: current index in candidates
    # used_primes: set of prime factors used by selected numbers
    # current_sum: sum of values of selected numbers
    # current_set: list of selected numbers
    
    def search(idx, used_primes, current_sum, current_set):
        nonlocal best_set, best_val, best_used_primes
        
        if time.time() > deadline - 5:
            return
            
        if idx == len(candidates):
            if current_sum > best_val:
                best_val = current_sum
                best_set = list(current_set)
                best_used_primes = set(used_primes)
            return
        
        # Upper bound calculation
        # Sum of values of all remaining candidates that are coprime to current selection
        # This is a valid upper bound because we assume we can pick all of them
        ub = current_sum
        for j in range(idx, len(candidates)):
            if factors[j].isdisjoint(used_primes):
                ub += vals[j]
                
        if ub <= best_val:
            return
            
        # Option 1: Skip candidates[idx]
        search(idx + 1, used_primes, current_sum, current_set)
        
        # Option 2: Pick candidates[idx]
        a = candidates[idx]
        if factors[a].isdisjoint(used_primes):
            new_used = used_primes | factors[a]
            current_set.append(a)
            search(idx + 1, new_used, current_sum + vals[a], current_set)
            current_set.pop()
            
    search(0, set(), 0.0, [])
    
    return best_set
