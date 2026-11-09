# EVOLVE-BLOCK-START
from math import prod, gcd, isqrt
import itertools

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

def _score(n):
    """Compute tau_perp(n) directly."""
    # Factorize n
    temp = n
    primes = []
    exps = []
    for p in _SMALL_PRIMES:
        if p * p > temp:
            break
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            primes.append(p)
            exps.append(e)
    if temp > 1:
        primes.append(temp)
        exps.append(1)
    
    divs = _divisors_sorted(primes, exps)
    return _tau_perp(divs)

def _best_square_free():
    """Find best square-free n with k primes, maximizing tau_perp."""
    best_score = 0
    best_n = 2
    
    # Try k from 15 down to 10
    for k in range(15, 9, -1):
        if k > len(_SMALL_PRIMES):
            continue
        # Use smallest k primes
        primes = _SMALL_PRIMES[:k]
        n = prod(primes)
        if n > _LIMIT:
            continue
        
        divs = _divisors_sorted(primes, [1] * k)
        s = _tau_perp(divs)
        
        if s > best_score:
            best_score = s
            best_n = n
            
        # Also try replacing the largest prime with a larger one to see if it helps
        # (unlikely to help for square-free, but check a few)
        for replace_idx in [k - 1]:
            base_n = n // primes[replace_idx]
            # Try some larger primes
            for q in _SMALL_PRIMES[k:k + 20]:
                if base_n * q > _LIMIT:
                    break
                new_primes = primes[:replace_idx] + [q]
                new_n = base_n * q
                divs2 = _divisors_sorted(new_primes, [1] * k)
                s2 = _tau_perp(divs2)
                if s2 > best_score:
                    best_score = s2
                    best_n = new_n
                    
    return best_n, best_score

def _best_with_exponents():
    """Try some structures with exponents > 1."""
    best_score = 0
    best_n = 2
    
    # Try k=14 with one exponent 2
    for k in range(13, 9, -1):
        if k > len(_SMALL_PRIMES):
            continue
        
        # Try all-1s
        primes = _SMALL_PRIMES[:k]
        n = prod(primes)
        if n <= _LIMIT:
            divs = _divisors_sorted(primes, [1] * k)
            s = _tau_perp(divs)
            if s > best_score:
                best_score = s
                best_n = n
        
        # Try one exponent 2, rest 1s
        for pos in range(k):
            exps = [1] * k
            exps[pos] = 2
            # Assign largest exponent to smallest prime to minimize product
            primes_sorted = _SMALL_PRIMES[:k]
            # Put exponent 2 on smallest prime
            exps2 = [1] * k
            exps2[0] = 2
            n2 = prod(p ** e for p, e in zip(primes_sorted, exps2))
            if n2 <= _LIMIT:
                divs2 = _divisors_sorted(primes_sorted, exps2)
                s2 = _tau_perp(divs2)
                if s2 > best_score:
                    best_score = s2
                    best_n = n2
                    
    return best_n, best_score

def construct():
    # Try square-free first (usually best)
    n_sq, s_sq = _best_square_free()
    
    # Try with exponents
    n_exp, s_exp = _best_with_exponents()
    
    # Pick the best
    if s_sq >= s_exp:
        return n_sq
    else:
        return n_exp
# EVOLVE-BLOCK-END
