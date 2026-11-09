import math
import time
from itertools import combinations

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number (product of distinct primes)."""
    divisors = [1]
    for p in primes:
        new_divs = [d * p for d in divisors]
        divisors.extend(new_divs)
    divisors.sort()
    return divisors

def _tau_perp_squarefree(primes):
    """Count adjacent coprime pairs among divisors of a squarefree number."""
    divs = _divisors_of_squarefree(primes)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _score_primes(prime_list):
    """Score a list of primes as a squarefree number."""
    n = 1
    for p in prime_list:
        n *= p
    if n > 10**18:
        return -1, n
    return _tau_perp_squarefree(prime_list), n

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Primes up to 97
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
    
    # Phase 2: Single swap - replace one prime in a base set with a larger one
    # Base sets: first 14 and first 15 primes
    for base_size in [14, 15]:
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
                new_n = base_n // base[i] * all_primes[j]
                if new_n > 10**18:
                    continue
                candidate = base[:i] + [all_primes[j]] + base[i+1:]
                candidate.sort()
                try_candidate(candidate)
    
    # Phase 3: Double swap - replace two primes
    for base_size in [14, 15]:
        base = all_primes[:base_size]
        base_n = 1
        for p in base:
            base_n *= p
        if base_n > 10**18:
            continue
        
        for i in range(base_size):
            for j in range(i + 1, base_size):
                for k in range(base_size, len(all_primes)):
                    for l in range(k + 1, len(all_primes)):
                        if time.time() > deadline:
                            return best_n
                        new_n = base_n // base[i] // base[j] * all_primes[k] * all_primes[l]
                        if new_n > 10**18:
                            continue
                        candidate = [p for idx, p in enumerate(base) if idx != i and idx != j]
                        candidate.append(all_primes[k])
                        candidate.append(all_primes[l])
                        candidate.sort()
                        try_candidate(candidate)
    
    # Phase 4: Try removing one prime from 15-prime set and adding a larger one (net 15 primes)
    base15 = all_primes[:15]
    base15_n = 1
    for p in base15:
        base15_n *= p
    
    if base15_n <= 10**18:
        for remove_idx in range(15):
            for add_idx in range(15, len(all_primes)):
                if time.time() > deadline:
                    return best_n
                new_n = base15_n // base15[remove_idx] * all_primes[add_idx]
                if new_n > 10**18:
                    continue
                candidate = [p for idx, p in enumerate(base15) if idx != remove_idx]
                candidate.append(all_primes[add_idx])
                candidate.sort()
                try_candidate(candidate)
    
    # Phase 5: Triple swap on 14-prime base
    base14 = all_primes[:14]
    base14_n = 1
    for p in base14:
        base14_n *= p
    
    if base14_n <= 10**18:
        for i in range(14):
            for j in range(i + 1, 14):
                for k in range(j + 1, 14):
                    for a in range(14, min(18, len(all_primes))):
                        for b in range(a + 1, min(20, len(all_primes))):
                            for c in range(b + 1, min(22, len(all_primes))):
                                if time.time() > deadline:
                                    return best_n
                                new_n = base14_n // base14[i] // base14[j] // base14[k] * all_primes[a] * all_primes[b] * all_primes[c]
                                if new_n > 10**18:
                                    continue
                                candidate = [p for idx, p in enumerate(base14) if idx not in (i, j, k)]
                                candidate.extend([all_primes[a], all_primes[b], all_primes[c]])
                                candidate.sort()
                                try_candidate(candidate)
    
    # Phase 6: Local search - perturb best solution
    # Find the prime list for best_n by factoring
    def factorize(n):
        factors = []
        temp = n
        for p in all_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.append(temp)
        return factors
    
    if best_n > 2:
        best_primes = factorize(best_n)
        # Try single swaps on the best solution
        used = set(best_primes)
        for i in range(len(best_primes)):
            for j in range(len(all_primes)):
                if time.time() > deadline:
                    return best_n
                if all_primes[j] in used:
                    continue
                new_n = best_n // best_primes[i] * all_primes[j]
                if new_n > 10**18:
                    continue
                candidate = best_primes[:i] + [all_primes[j]] + best_primes[i+1:]
                candidate.sort()
                try_candidate(candidate)
    
    return best_n
