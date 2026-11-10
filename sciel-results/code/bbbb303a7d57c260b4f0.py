import time
import random

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True

def _build_max_set():
    primes = [p for p in range(2, 41) if _is_prime(p)]
    composites = [c for c in range(2, 41) if not _is_prime(c)]
    
    best_set = list(primes)
    
    deadline = time.time() + 80
    max_found = len(primes)
    best_candidate = list(primes)
    
    for _ in range(500):
        if time.time() > deadline:
            break
            
        S = list(primes)
        # Maintain the set of all subset products incrementally
        prods = {1}
        for a in S:
            new_prods = {x * a for x in prods}
            prods |= new_prods
        
        random.shuffle(composites)
        
        for c in composites:
            if len(S) >= 18:
                break
            # Check if adding c creates a collision
            new_prods = {x * c for x in prods}
            if len(new_prods & prods) == 0:
                # No collision - add c
                S.append(c)
                prods |= new_prods
                
        if len(S) > max_found:
            max_found = len(S)
            best_candidate = list(S)
            if max_found == 18:
                break
                
    return best_candidate

def construct():
    t0 = time.time()
    S = _build_max_set()
    
    # Ensure list of Python ints
    S = [int(x) for x in S]
    
    # Trim to max 18 if needed
    if len(S) > 18:
        S = S[:18]
        
    # Final verification with time guard
    if time.time() - t0 < 85:
        prods = [1]
        for a in S:
            prods = prods + [x * a for x in prods]
        if len(prods) != len(set(prods)):
            # Fallback: return just primes
            S = [p for p in range(2, 41) if _is_prime(p)]
            
    return S
