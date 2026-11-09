import math
import time
from itertools import combinations

def _divisors_bitmasks(primes):
    """Generate divisors as (value, bitmask) pairs for a squarefree number.
    Bitmask has bit i set if prime i divides the divisor."""
    k = len(primes)
    divs = [(1, 0)]
    for i, p in enumerate(primes):
        bit = 1 << i
        new_divs = [(v * p, m | bit) for v, m in divs]
        divs.extend(new_divs)
    divs.sort(key=lambda x: x[0])
    return divs

def _tau_perp_bitmask(divs):
    """Count adjacent coprime pairs using bitmask representation.
    Two divisors are coprime iff their bitmasks are disjoint (AND == 0)."""
    count = 0
    for i in range(len(divs) - 1):
        if (divs[i][1] & divs[i + 1][1]) == 0:
            count += 1
    return count

def _score_primes(prime_list):
    """Score a list of primes as a squarefree number using bitmask method."""
    n = 1
    for p in prime_list:
        n *= p
        if n > 10**18:
            return -1, n
    divs = _divisors_bitmasks(prime_list)
    return _tau_perp_bitmask(divs), n

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Primes up to ~100 for search space
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_score = -1
    best_n = 2
    
    def try_candidate(prime_list):
        nonlocal best_score, best_n
        if time.time() > deadline:
            return False
        s, n = _score_primes(prime_list)
        if n <= 10**18 and s > best_score:
            best_score = s
            best_n = n
        return True
    
    # Phase 1: Try primorials of first k primes for k=2..15
    for k in range(2, 16):
        primes_k = all_primes[:k]
        n_prod = 1
        for p in primes_k:
            n_prod *= p
        if n_prod > 10**18:
            break
        try_candidate(primes_k)
    
    # Phase 2: Systematic single swap search
    # For each base size (13-15), try replacing one prime with a larger unused one
    for base_size in [13, 14, 15]:
        if time.time() > deadline - 20:
            break
        base = all_primes[:base_size]
        base_n = 1
        for p in base:
            base_n *= p
        if base_n > 10**18:
            continue
        
        used = set(base)
        for i in range(base_size):
            for j in range(base_size, len(all_primes)):
                if all_primes[j] in used:
                    continue
                new_primes = base[:i] + [all_primes[j]] + base[i+1:]
                new_n = base_n // base[i] * all_primes[j]
                if new_n > 10**18:
                    continue
                try_candidate(new_primes)
                if time.time() > deadline - 15:
                    break
            if time.time() > deadline - 15:
                break
    
    # Phase 3: Double swap search for best base sizes
    for base_size in [14, 15]:
        if time.time() > deadline - 20:
            break
        base = all_primes[:base_size]
        base_n = 1
        for p in base:
            base_n *= p
        if base_n > 10**18:
            continue
        
        used = set(base)
        for i in range(base_size):
            for j in range(i + 1, base_size):
                for k in range(base_size, min(len(all_primes), base_size + 12)):
                    for l in range(k + 1, min(len(all_primes), base_size + 12)):
                        if all_primes[k] in used or all_primes[l] in used:
                            continue
                        new_primes = base[:i] + [all_primes[k]] + base[i+1:j] + [all_primes[l]] + base[j+1:]
                        new_n = base_n // base[i] // base[j] * all_primes[k] * all_primes[l]
                        if new_n > 10**18:
                            continue
                        try_candidate(new_primes)
                        if time.time() > deadline - 10:
                            break
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
    
    # Phase 4: Broader combination search for sizes 13-15 from first 20 primes
    for k in [13, 14, 15]:
        if time.time() > deadline - 25:
            break
        # Try combinations from first 20 primes
        for subset in combinations(all_primes[:20], k):
            n_prod = 1
            for p in subset:
                n_prod *= p
                if n_prod > 10**18:
                    break
            else:
                try_candidate(list(subset))
                if time.time() > deadline - 20:
                    break
    
    # Phase 5: Local refinement - try triple swaps from best candidate
    if time.time() < deadline - 30 and best_score > 0:
        # Try to improve by replacing up to 3 primes in the best candidate
        # First, factor best_n to get its primes
        best_n_temp = best_n
        best_primes = []
        for p in all_primes:
            if best_n_temp % p == 0:
                best_primes.append(p)
                best_n_temp //= p
        if len(best_primes) > 0 and best_n_temp == 1:
            # Try single swaps on the best candidate
            for i in range(len(best_primes)):
                for j in range(len(all_primes)):
                    if all_primes[j] in best_primes:
                        continue
                    new_primes = best_primes[:i] + [all_primes[j]] + best_primes[i+1:]
                    new_n = best_n // best_primes[i] * all_primes[j]
                    if new_n > 10**18:
                        continue
                    try_candidate(new_primes)
                    if time.time() > deadline - 20:
                        break
                if time.time() > deadline - 20:
                    break
    
    return best_n
