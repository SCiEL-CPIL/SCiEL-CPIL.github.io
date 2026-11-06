import time
import sys
import random

def is_valid(seq):
    """Check if all consecutive block products are distinct."""
    if not seq:
        return True
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def is_valid_fast(seq):
    """Faster validity check using early termination and set operations."""
    if not seq:
        return True
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def get_primes_up_to(n):
    """Get all primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def try_insert_composite(seq, composites, deadline, rng):
    """Try to insert composites one at a time, in random order."""
    selected = list(seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle composites not already in selected
        available = [c for c in composites if c not in selected]
        rng.shuffle(available)
        for x in available:
            if time.time() > deadline:
                break
            candidate = selected + [x]
            candidate.sort()
            if is_valid_fast(candidate):
                selected = candidate
                improved = True
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Get all primes in [2, 128] - these form a valid base
    primes = get_primes_up_to(128)
    composites = [x for x in range(2, 129) if x not in primes]
    
    best = list(primes)
    best_len = len(best)
    
    # Multiple restarts with different random seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        
        # Start with primes, then try to insert composites
        current = list(primes)
        
        # Try inserting composites in random order
        current = try_insert_composite(current, composites, deadline, rng)
        
        if len(current) > best_len:
            best = current
            best_len = len(current)
    
    # Final pass: try to add any remaining numbers in ascending order
    if time.time() < deadline:
        for x in range(2, 129):
            if time.time() > deadline:
                break
            if x in best:
                continue
            candidate = best + [x]
            candidate.sort()
            if is_valid_fast(candidate):
                best = candidate
    
    return best
