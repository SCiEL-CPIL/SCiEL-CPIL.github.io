import math
import time
from math import gcd

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_up_to(60)

def _tau_perp(primes, exponents):
    """Compute tau_perp score from prime factorization."""
    divisors = [1]
    for p, e in zip(primes, exponents):
        if e == 0:
            continue
        current = list(divisors)
        pk = 1
        for k in range(1, e + 1):
            pk *= p
            for d in current:
                divisors.append(d * pk)
    divisors.sort()
    if len(divisors) < 2:
        return 0
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _compute_n(primes, exponents):
    n = 1
    for p, e in zip(primes, exponents):
        if e > 0:
            n *= p ** e
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    max_n = 10**18
    primes = _PRIMES
    
    # Determine how many primes we can use with all exponent 1
    prod = 1
    max_k = 0
    for i, p in enumerate(primes):
        if prod * p <= max_n:
            prod *= p
            max_k = i + 1
        else:
            break
    
    # Initial: primorial of first max_k primes
    current_exps = [1] * max_k
    current_n = _compute_n(primes, current_exps)
    current_score = _tau_perp(primes, current_exps)
    
    best_score = current_score
    best_n = current_n
    best_exps = current_exps[:]
    
    # Local search: try changing one exponent at a time
    # For each prime, try increasing or decreasing its exponent
    improved = True
    max_iter = 0
    
    while improved and time.time() < deadline:
        improved = False
        max_iter += 1
        
        for i in range(len(current_exps)):
            if time.time() > deadline:
                break
            
            p = primes[i]
            
            # Try increasing exponent
            if current_exps[i] < 10:
                new_exps = current_exps[:]
                new_exps[i] += 1
                new_n = _compute_n(primes, new_exps)
                if new_n <= max_n:
                    new_score = _tau_perp(primes, new_exps)
                    if new_score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_exps = current_exps[:]
            
            # Try decreasing exponent (if > 0)
            if current_exps[i] > 0:
                new_exps = current_exps[:]
                new_exps[i] -= 1
                new_n = _compute_n(primes, new_exps)
                if new_n <= max_n:
                    new_score = _tau_perp(primes, new_exps)
                    if new_score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_exps = current_exps[:]
        
        # Also try: add a new prime (if we haven't used all max_k)
        if len(current_exps) < max_k:
            if time.time() > deadline:
                break
            new_exps = current_exps + [1]
            new_n = _compute_n(primes, new_exps)
            if new_n <= max_n:
                new_score = _tau_perp(primes, new_exps)
                if new_score > current_score:
                    current_exps = new_exps
                    current_n = new_n
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
                        best_exps = current_exps[:]
    
    # Also try: remove a prime and add a different one
    # This is a 2-move local search
    if time.time() < deadline:
        for i in range(len(current_exps)):
            if time.time() > deadline:
                break
            if current_exps[i] == 0:
                continue
            # Remove prime i
            temp_exps = current_exps[:]
            temp_exps[i] = 0
            temp_n = _compute_n(primes, temp_exps)
            
            # Try adding each other prime with exponent 1
            for j in range(max_k):
                if j == i:
                    continue
                if temp_exps[j] > 0:
                    continue
                new_exps = temp_exps[:]
                new_exps[j] = 1
                new_n = temp_n * primes[j]
                if new_n <= max_n:
                    new_score = _tau_perp(primes, new_exps)
                    if new_score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_exps = current_exps[:]
    
    return best_n
