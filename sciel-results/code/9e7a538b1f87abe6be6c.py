# EVOLVE-BLOCK-START
from math import prod, gcd, isqrt
import itertools
from functools import lru_cache

_LIMIT = 10**18

# Small primes
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

def _divisors_sorted(primes, exps):
    """Generate all divisors of n = prod(p_i ** e_i) in sorted order."""
    divs = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        p_power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * p_power)
            p_power *= p
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(divs):
    """Count adjacent coprime pairs in sorted divisor list."""
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _score_from_factors(primes, exps):
    """Compute tau_perp from prime factorization without recomputing divisors if possible."""
    divs = _divisors_sorted(primes, exps)
    return _tau_perp(divs)

def _search_square_free():
    """Exhaustive search over square-free numbers with up to 16 primes."""
    best_score = 0
    best_n = 2
    
    # For square-free, the number of divisors is 2^k
    # We want to maximize coprime adjacent pairs
    # Try all combinations of k primes from the first ~40 primes
    # But that's too many. Let's be smarter.
    
    # Key insight: for square-free n, two divisors are coprime iff their
    # prime sets are disjoint. In the sorted order, we need to count adjacent
    # pairs with disjoint prime sets.
    
    # For the smallest primes, let's just try k=14,15,16 with smallest primes
    # and also try some variations
    
    for k in range(16, 10, -1):
        # Start with smallest k primes
        base_primes = _SMALL_PRIMES[:k]
        base_n = prod(base_primes)
        if base_n > _LIMIT:
            continue
        
        s = _score_from_factors(base_primes, [1]*k)
        if s > best_score:
            best_score = s
            best_n = base_n
        
        # Try replacing the largest prime with slightly larger primes
        for i in range(k-1, max(k-5, -1), -1):
            base_without = base_n // base_primes[i]
            for j in range(k, min(k+15, len(_SMALL_PRIMES))):
                q = _SMALL_PRIMES[j]
                if q <= base_primes[i]:
                    continue
                new_n = base_without * q
                if new_n > _LIMIT:
                    break
                new_primes = base_primes[:i] + [q] + base_primes[i+1:]
                new_primes.sort()
                s2 = _score_from_factors(new_primes, [1]*k)
                if s2 > best_score:
                    best_score = s2
                    best_n = new_n
    
    return best_n, best_score

def _search_with_exponents():
    """Search over numbers with some exponents > 1."""
    best_score = 0
    best_n = 2
    
    # Try structures like p1^2 * p2 * p3 * ... * pk
    # The exponent on the smallest prime is best to keep product low
    
    for k in range(15, 10, -1):
        # Try p1^2 * p2 * ... * pk
        primes = _SMALL_PRIMES[:k]
        exps = [2] + [1]*(k-1)
        n = prod(p**e for p, e in zip(primes, exps))
        if n <= _LIMIT:
            s = _score_from_factors(primes, exps)
            if s > best_score:
                best_score = s
                best_n = n
        
        # Try p1 * p2^2 * p3 * ... * pk (exponent on second prime)
        exps2 = [1, 2] + [1]*(k-2)
        n2 = prod(p**e for p, e in zip(primes, exps2))
        if n2 <= _LIMIT:
            s2 = _score_from_factors(primes, exps2)
            if s2 > best_score:
                best_score = s2
                best_n = n2
        
        # Try p1^3 * p2 * ... * pk
        exps3 = [3] + [1]*(k-1)
        n3 = prod(p**e for p, e in zip(primes, exps3))
        if n3 <= _LIMIT:
            s3 = _score_from_factors(primes, exps3)
            if s3 > best_score:
                best_score = s3
                best_n = n3
        
        # Try p1^2 * p2^2 * p3 * ... * pk
        if k >= 2:
            exps4 = [2, 2] + [1]*(k-2)
            n4 = prod(p**e for p, e in zip(primes, exps4))
            if n4 <= _LIMIT:
                s4 = _score_from_factors(primes, exps4)
                if s4 > best_score:
                    best_score = s4
                    best_n = n4
    
    return best_n, best_score

def _search_mixed():
    """Try mixed structures with different prime sets."""
    best_score = 0
    best_n = 2
    
    # Try skipping some small primes and using larger ones
    # e.g., skip 2, use 3,5,7,11,...
    for skip in [2, 3, 5]:
        if skip > len(_SMALL_PRIMES):
            continue
        for k in range(15, 11, -1):
            # Use primes starting from index skip
            primes = _SMALL_PRIMES[skip:skip+k]
            if len(primes) < k:
                continue
            n = prod(primes)
            if n > _LIMIT:
                continue
            s = _score_from_factors(primes, [1]*k)
            if s > best_score:
                best_score = s
                best_n = n
    
    return best_n, best_score

def construct():
    # Run multiple search strategies
    n1, s1 = _search_square_free()
    n2, s2 = _search_with_exponents()
    n3, s3 = _search_mixed()
    
    # Pick the best
    candidates = [(n1, s1), (n2, s2), (n3, s3)]
    best_n, best_s = max(candidates, key=lambda x: x[1])
    return best_n
# EVOLVE-BLOCK-END
