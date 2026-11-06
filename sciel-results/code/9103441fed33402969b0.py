import math
import time

def score_from_factors(factors):
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        p_power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * p_power)
            p_power *= p
        divs = new_divs
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def construct():
    # Primes up to 300
    def gen_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = gen_primes(300)
    
    # Start with P15
    start_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    best_factors = {p: 1 for p in start_primes}
    best_n = 1
    for p in start_primes:
        best_n *= p
        
    best_score = score_from_factors(best_factors)
    
    current_factors = best_factors.copy()
    current_n = best_n
    current_score = best_score
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Local search
    while time.time() < deadline:
        improved = False
        
        # 1. Increase exponent
        for p in list(current_factors.keys()):
            if current_n * p <= 10**18:
                new_factors = current_factors.copy()
                new_factors[p] += 1
                new_n = current_n * p
                s = score_from_factors(new_factors)
                if s > current_score:
                    current_factors = new_factors
                    current_n = new_n
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_factors = current_factors.copy()
                        best_n = current_n
                    break
        
        # 2. Decrease exponent
        if not improved:
            for p in list(current_factors.keys()):
                if current_factors[p] > 1:
                    new_factors = current_factors.copy()
                    new_factors[p] -= 1
                    new_n = current_n // p
                    s = score_from_factors(new_factors)
                    if s > current_score:
                        current_factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_factors = current_factors.copy()
                            best_n = current_n
                        break

        # 3. Remove prime
        if not improved:
            for p in list(current_factors.keys()):
                if current_factors[p] == 1:
                    new_factors = current_factors.copy()
                    del new_factors[p]
                    new_n = current_n // p
                    s = score_from_factors(new_factors)
                    if s > current_score:
                        current_factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_factors = current_factors.copy()
                            best_n = current_n
                        break

        # 4. Add prime
        if not improved:
            for q in primes:
                if q not in current_factors:
                    if current_n * q <= 10**18:
                        new_factors = current_factors.copy()
                        new_factors[q] = 1
                        new_n = current_n * q
                        s = score_from_factors(new_factors)
                        if s > current_score:
                            current_factors = new_factors
                            current_n = new_n
                            current_score = s
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_factors = current_factors.copy()
                                best_n = current_n
                            break

        # 5. Replace prime
        if not improved:
            for p in list(current_factors.keys()):
                if current_factors[p] == 1:
                    for q in primes:
                        if q == p or q in current_factors:
                            continue
                        new_n = current_n // p * q
                        if new_n > 10**18:
                            continue
                        new_factors = current_factors.copy()
                        del new_factors[p]
                        new_factors[q] = 1
                        s = score_from_factors(new_factors)
                        if s > current_score:
                            current_factors = new_factors
                            current_n = new_n
                            current_score = s
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_factors = current_factors.copy()
                                best_n = current_n
                            break
                    if improved:
                        break

        if not improved:
            break

    return best_n
