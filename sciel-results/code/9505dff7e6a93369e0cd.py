"""Search for squarefree n maximizing coprime adjacent divisor pairs."""
# EVOLVE-BLOCK-START
import time
import math
import itertools

DEADLINE = 140.0
MAX_N = 10**18

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

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
    
    max_k = 15  # max number of primes
    
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
    
    # Now try single and double swaps for k in range where it makes sense
    for k in range(10, min(max_k + 1, 16)):
        if time.time() - start > DEADLINE - 10:
            break
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
            for j in range(k, min(len(_SMALL_PRIMES), k + 15)):
                if _SMALL_PRIMES[j] in used:
                    continue
                new_primes = base[:i] + [_SMALL_PRIMES[j]] + base[i+1:]
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
        
        # Try double swaps: replace two primes with two larger unused ones
        if time.time() - start > DEADLINE - 10:
            continue
        for i in range(k):
            for j in range(i+1, k):
                if time.time() - start > DEADLINE - 10:
                    break
                used2 = set(base)
                for a in range(k, min(len(_SMALL_PRIMES), k + 10)):
                    if _SMALL_PRIMES[a] in used2:
                        continue
                    for b in range(a+1, min(len(_SMALL_PRIMES), k + 10)):
                        if _SMALL_PRIMES[b] in used2:
                            continue
                        new_primes = base[:i] + [_SMALL_PRIMES[a]] + base[i+1:j] + [_SMALL_PRIMES[b]] + base[j+1:]
                        new_n = base_n // base[i] // base[j] * _SMALL_PRIMES[a] * _SMALL_PRIMES[b]
                        if new_n > MAX_N:
                            continue
                        s3, _ = _score_primes(new_primes)
                        if s3 > best_score:
                            best_score = s3
                            best_n = new_n
                        if time.time() - start > DEADLINE - 10:
                            break
                    if time.time() - start > DEADLINE - 10:
                        break
            if time.time() - start > DEADLINE - 10:
                break
    
    # Try a few more targeted searches for k=14,15
    for k in [14, 15]:
        if time.time() - start > DEADLINE - 15:
            break
        candidates = [
            _SMALL_PRIMES[:k],
        ]
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
