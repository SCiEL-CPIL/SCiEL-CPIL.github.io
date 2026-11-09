"""Search for squarefree n maximizing coprime adjacent divisor pairs."""
# EVOLVE-BLOCK-START
import time
import math
import itertools

DEADLINE = 140.0
MAX_N = 10**18

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number given its prime list."""
    divs = [1]
    for p in primes:
        divs = divs + [d * p for d in divs]
    divs.sort()
    return divs

def _tau_perp(divs):
    """Count adjacent coprime pairs in sorted divisor list."""
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def _score_primes(primes):
    """Score for squarefree number with given primes."""
    n = 1
    for p in primes:
        n *= p
    if n > MAX_N:
        return -1, n
    divs = _divisors_of_squarefree(primes)
    return _tau_perp(divs), n

def construct():
    start = time.time()
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # seed value
    
    # First, evaluate the seed
    seed_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    s, n = _score_primes(seed_primes)
    best_score = s
    best_n = n
    
    # Try all subsets of small primes (up to size ~15-16 where product stays under 10^18)
    # The product of first 15 primes: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    # Let's compute: primorial(47) ≈ 6.1e17, primorial(53) ≈ 3.2e19 > 10^18
    # So we can use up to 15 smallest primes (product ~6.1e17) or swap some for larger ones
    
    # Evaluate all subsets of the first ~18 primes, but only those with product <= 10^18
    # For efficiency, iterate by subset size
    
    max_k = 15  # max number of primes
    
    # Precompute: for each subset size, try combinations
    # Start with the full set of smallest primes, then try replacements
    
    # Strategy: try all combinations of up to max_k primes from _SMALL_PRIMES
    # but prune by product
    
    # First try the greedy: smallest k primes for each k
    for k in range(2, max_k + 1):
        if time.time() - start > DEADLINE - 10:
            break
        primes = _SMALL_PRIMES[:k]
        n = 1
        for p in primes:
            n *= p
            if n > MAX_N:
                break
        else:
            s, _ = _score_primes(primes)
            if s > best_score:
                best_score = s
                best_n = n
            continue
        # If product too big, skip this k
    
    # Now try swapping: replace some larger primes with smaller unused ones
    # Actually, the smallest primes give the most divisors for a given count,
    # but the score depends on the actual values (ordering matters)
    
    # Let's do a more thorough search: try all subsets of size 10-15 from first 20 primes
    # where product <= 10^18
    
    # For k in range where it makes sense:
    for k in range(10, min(max_k + 1, 16)):
        if time.time() - start > DEADLINE - 10:
            break
        # Try combinations from first 20 primes, but only those with product <= MAX_N
        # To be efficient, start with smallest and do local swaps
        base = _SMALL_PRIMES[:k]
        base_n = 1
        for p in base:
            base_n *= p
        
        if base_n > MAX_N:
            continue
        
        s, _ = _score_primes(base)
        if s > best_score:
            best_score = s
            best_n = base_n
        
        # Try single swaps: replace one prime with a larger unused one
        used = set(base)
        for i in range(k):
            for j in range(k, min(len(_SMALL_PRIMES), k + 10)):
                if _SMALL_PRIMES[j] in used:
                    continue
                new_primes = base[:i] + [_SMALL_PRIMES[j]] + base[i+1:]
                # Check product
                new_n = base_n // base[i] * _SMALL_PRIMES[j]
                if new_n > MAX_N:
                    continue
                s2, _ = _score_primes(new_primes)
                if s2 > best_score:
                    best_score = s2
                    best_n = new_n
                if time.time() - start > DEADLINE - 10:
                    break
            if time.time() - start > DEADLINE - 10:
                break
    
    # Also try: what if we use primes that aren't the smallest?
    # E.g., skip 2 and use more odd primes? Unlikely to help since 2 is small.
    
    # Try a few more targeted searches
    # What about k=14,15 with various combinations?
    for k in [14, 15]:
        if time.time() - start > DEADLINE - 15:
            break
        # Try a few specific combinations
        candidates = [
            _SMALL_PRIMES[:k],
        ]
        # Try replacing the largest with next available
        if k < len(_SMALL_PRIMES):
            c2 = _SMALL_PRIMES[:k-1] + [_SMALL_PRIMES[k]]
            candidates.append(c2)
        
        for cand in candidates:
            n = 1
            for p in cand:
                n *= p
            if n > MAX_N:
                continue
            s, _ = _score_primes(cand)
            if s > best_score:
                best_score = s
                best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
