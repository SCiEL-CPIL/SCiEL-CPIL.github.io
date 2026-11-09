"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from math import gcd
from itertools import product

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        p_power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * p_power)
            p_power *= p
        divs = new_divs
    return sorted(divs)

def _tau_perp(n):
    """Count adjacent divisor pairs with gcd 1."""
    if n < 2:
        return 0
    # Factorize
    factors = {}
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    
    divs = _divisors_from_factors(factors)
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _tau_perp_from_factors(factors):
    """Count adjacent divisor pairs with gcd 1 from factorization."""
    divs = _divisors_from_factors(factors)
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _primes_below(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = _tau_perp(best_n)
    
    small_primes = _primes_below(200)
    
    # Try squarefree products of various prime sets
    for k in range(2, 16):
        if time.time() > deadline:
            break
        n = 1
        for p in small_primes[:k]:
            n *= p
        if n > 10**18:
            break
        score = _tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Try prime powers
    for p in [2, 3, 5, 7, 11, 13]:
        if time.time() > deadline:
            break
        n = 1
        for a in range(1, 60):
            n *= p
            if n > 10**18:
                break
            score = _tau_perp(n)
            if score > best_score:
                best_score = score
                best_n = n
    
    # Systematic search over small prime sets with exponents
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [3, 5, 7, 11],
        [5, 7, 11, 13],
    ]
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
        max_exps = []
        for p in primes:
            max_e = 0
            prod = 1
            while prod * p <= 10**18:
                prod *= p
                max_e += 1
            max_exps.append(max_e)
        
        total_combos = 1
        for me in max_exps:
            total_combos *= (me + 1)
        
        if total_combos < 200000:
            ranges = [range(me + 1) for me in max_exps]
            for exponents in product(*ranges):
                if time.time() > deadline:
                    break
                n = 1
                for p, e in zip(primes, exponents):
                    if e > 0:
                        n *= p ** e
                if n < 2 or n > 10**18:
                    continue
                score = _tau_perp(n)
                if score > best_score:
                    best_score = score
                    best_n = n
        else:
            random.seed(42)
            for _ in range(30000):
                if time.time() > deadline:
                    break
                exponents = [random.randint(0, me) for me in max_exps]
                n = 1
                for p, e in zip(primes, exponents):
                    if e > 0:
                        n *= p ** e
                if n < 2 or n > 10**18:
                    continue
                score = _tau_perp(n)
                if score > best_score:
                    best_score = score
                    best_n = n
    
    # Local search
    if time.time() < deadline:
        factors = {}
        temp = best_n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        
        primes_list = sorted(factors.keys())
        
        for _ in range(30000):
            if time.time() > deadline:
                break
            n = best_n
            
            # Try increasing a random prime's exponent
            p = random.choice(primes_list)
            current_exp = factors.get(p, 0)
            
            if current_exp < 60:
                new_n = n * p
                if new_n <= 10**18:
                    score = _tau_perp(new_n)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
                        factors = {}
                        temp = new_n
                        d = 2
                        while d * d <= temp:
                            while temp % d == 0:
                                factors[d] = factors.get(d, 0) + 1
                                temp //= d
                            d += 1
                        if temp > 1:
                            factors[temp] = factors.get(temp, 0) + 1
                        primes_list = sorted(factors.keys())
            
            # Try decreasing
            if current_exp > 0:
                new_n = n // p
                if new_n >= 2:
                    score = _tau_perp(new_n)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
                        factors = {}
                        temp = new_n
                        d = 2
                        while d * d <= temp:
                            while temp % d == 0:
                                factors[d] = factors.get(d, 0) + 1
                                temp //= d
                            d += 1
                        if temp > 1:
                            factors[temp] = factors.get(temp, 0) + 1
                        primes_list = sorted(factors.keys())
            
            # Try adding a new small prime
            candidates = [q for q in small_primes if q not in factors]
            if candidates:
                new_p = random.choice(candidates)
                new_n = n * new_p
                if new_n <= 10**18:
                    score = _tau_perp(new_n)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
                        factors = {}
                        temp = new_n
                        d = 2
                        while d * d <= temp:
                            while temp % d == 0:
                                factors[d] = factors.get(d, 0) + 1
                                temp //= d
                            d += 1
                        if temp > 1:
                            factors[temp] = factors.get(temp, 0) + 1
                        primes_list = sorted(factors.keys())
    
    return best_n
# EVOLVE-BLOCK-END
