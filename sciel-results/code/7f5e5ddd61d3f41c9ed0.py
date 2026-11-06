import math
import random
import time

def _tau_perp(n):
    """Calculate tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    if n < 2:
        return 0
    
    # Factorize n
    temp = n
    factors = {}
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            factors[d] = count
        if d == 2:
            d = 3
        else:
            d += 2
    if temp > 1:
        factors[temp] = 1
        
    # Generate divisors
    divisors = [1]
    for p, e in factors.items():
        current_divs = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                current_divs.append(d * power)
            power *= p
        divisors = current_divs
        
    divisors.sort()
    
    score = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            score += 1
            
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial candidates
    # 1. Primorial of 15 primes
    p15 = 614889782588491410
    # 2. Highly composite number candidate
    hcn = 897612484786617600
    # 3. Another candidate: product of first 14 primes * 53
    p14 = p15 // 47
    cand3 = p14 * 53
    # 4. Product of first 13 primes * 53 * 59?
    p13 = p14 // 43
    cand4 = p13 * 53 * 59
    
    candidates = [p15, hcn, cand3, cand4]
    
    best_n = candidates[0]
    best_score = _tau_perp(best_n)
    
    # Local search
    def get_factorization(n):
        temp = n
        factors = {}
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                count = 0
                while temp % d == 0:
                    temp //= d
                    count += 1
                factors[d] = count
            if d == 2:
                d = 3
            else:
                d += 2
        if temp > 1:
            factors[temp] = 1
        return factors

    def build_from_factors(factors):
        n = 1
        for p, e in factors.items():
            n *= p ** e
        return n

    current_n = best_n
    current_factors = get_factorization(current_n)
    
    # Small primes for mutation
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    while time.time() < deadline:
        # Pick a random candidate to explore from (restart occasionally)
        if random.random() < 0.05:
            current_n = random.choice(candidates)
            current_factors = get_factorization(current_n)
            score = _tau_perp(current_n)
            if score > best_score:
                best_score = score
                best_n = current_n
            continue

        # Mutate current_factors
        new_factors = dict(current_factors)
        
        # Strategy: change exponent of a random prime, or add/remove a small prime
        action = random.random()
        
        if action < 0.4:
            # Increment exponent of a random prime in factors
            if new_factors:
                p = random.choice(list(new_factors.keys()))
                new_factors[p] += 1
            else:
                continue
        elif action < 0.8:
            # Decrement exponent of a random prime in factors
            if new_factors:
                p = random.choice(list(new_factors.keys()))
                if new_factors[p] > 1:
                    new_factors[p] -= 1
                else:
                    del new_factors[p]
            else:
                continue
        else:
            # Add or remove a random small prime
            p = random.choice(small_primes)
            if p in new_factors:
                if new_factors[p] > 1:
                    new_factors[p] -= 1
                else:
                    del new_factors[p]
            else:
                new_factors[p] = 1
        
        try:
            candidate_n = build_from_factors(new_factors)
        except OverflowError:
            continue
            
        if candidate_n > 10**18 or candidate_n < 2:
            continue
            
        score = _tau_perp(candidate_n)
        
        if score > best_score:
            best_score = score
            best_n = candidate_n
            current_n = candidate_n
            current_factors = new_factors
        elif score > _tau_perp(current_n) or random.random() < 0.1:
            # Accept if better than current or with small probability
            current_n = candidate_n
            current_factors = new_factors
            
    return best_n
