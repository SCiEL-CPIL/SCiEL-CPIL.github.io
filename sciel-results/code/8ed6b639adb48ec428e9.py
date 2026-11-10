"""Combined program: Prime base with greedy composite expansion."""
# EVOLVE-BLOCK-START
import time

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

def _has_distinct_subset_products(S):
    """Check that all 2^k subset products of S are distinct."""
    k = len(S)
    if k == 0:
        return True
    prods = [1]
    for a in S:
        new_prods = [x * a for x in prods]
        prods.extend(new_prods)
    return len(prods) == len(set(prods))

def _build_max_set():
    """
    1. Start with all primes <= 40.
    2. Greedily add composites in increasing order if valid.
    """
    primes = [p for p in range(2, 41) if _is_prime(p)]
    composites = [c for c in range(2, 41) if not _is_prime(c)]
    
    S = list(primes)
    
    # Verify initial set
    if not _has_distinct_subset_products(S):
        S = []
        
    # Greedily add composites
    for c in composites:
        if len(S) >= 18:
            break
        trial = S + [c]
        if _has_distinct_subset_products(trial):
            S = trial
            
    return S

def construct():
    """Return up to 18 distinct integers in 2..40 with all subset products distinct."""
    t0 = time.time()
    
    # Primary strategy: Primes + Greedy Composites
    S = _build_max_set()
    
    # Ensure list of Python ints
    S = [int(x) for x in S]
    
    # Trim to max 18 if needed (shouldn't happen with limit in loop)
    if len(S) > 18:
        S = S[:18]
        
    # Final verification
    if time.time() - t0 < 85:
        prods = [1]
        for a in S:
            prods = prods + [x * a for x in prods]
        if len(prods) != len(set(prods)):
            # Fallback: return just primes
            S = [p for p in range(2, 41) if _is_prime(p)]
            
    return S
# EVOLVE-BLOCK-END
