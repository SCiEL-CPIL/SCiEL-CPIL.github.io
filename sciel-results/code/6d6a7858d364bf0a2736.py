import math
import time

def _score_from_factorization(primes, exponents):
    """Compute tau_perp score from prime factorization."""
    divisors = [1]
    for p, a in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            val = d
            for e in range(a + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    divisors.sort()
    
    score = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            score += 1
    return score

def construct():
    deadline = time.time() + 140
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    candidates = []
    seen = set()
    
    def add_candidate(primes, exponents):
        n = 1
        for p, e in zip(primes, exponents):
            n *= p ** e
        if n > 10**18 or n < 2:
            return
        key = tuple(zip(primes, exponents))
        if key not in seen:
            seen.add(key)
            candidates.append((primes, exponents, n))
    
    # 1. Primorials
    for k in range(2, 16):
        if k > len(primes_list):
            break
        primes = primes_list[:k]
        exponents = [1] * k
        add_candidate(primes, exponents)
    
    # 2. For each small prime set, try non-increasing exponent patterns
    for m in range(2, 11):
        if m > len(primes_list):
            break
        base_primes = primes_list[:m]
        
        def gen_patterns(idx, max_exp, current, product):
            if time.time() > deadline:
                return
            if idx == m:
                if product <= 10**18 and product >= 2:
                    add_candidate(base_primes, current)
                return
            
            p = base_primes[idx]
            for e in range(0, max_exp + 1):
                new_prod = product * (p ** e)
                if new_prod > 10**18:
                    break
                gen_patterns(idx + 1, e, current + [e], new_prod)
        
        gen_patterns(0, 10, [], 1)
    
    # Evaluate all candidates
    for primes, exponents, n in candidates:
        if time.time() > deadline:
            break
        score = _score_from_factorization(primes, exponents)
        if score > best_score:
            best_score = score
            best_n = n
    
    return best_n
