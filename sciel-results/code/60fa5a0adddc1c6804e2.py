import time
import math
import random

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

def get_divisors(primes):
    divs = [1]
    for p in primes:
        new_divs = [d * p for d in divs]
        divs.extend(new_divs)
    return divs

def score_from_primes(primes):
    divs = get_divisors(primes)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def product(primes):
    p = 1
    for x in primes:
        p *= x
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate primes up to 500 for replacement candidates
    primes = sieve(500)
    
    # Start with the 15 smallest primes, which fit in 10^18
    base_primes = primes[:15]
    current_primes = base_primes[:]
    current_score = score_from_primes(current_primes)
    best_score = current_score
    best_primes = current_primes[:]
    
    # Systematic hill climbing with random restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            current_primes = base_primes[:]
        else:
            current_primes = best_primes[:]
            for _ in range(random.randint(1, 2)):
                i = random.randint(0, len(current_primes) - 1)
                p_old = current_primes[i]
                current_prod = product(current_primes)
                max_allowed_q = int((10**18 * p_old) / current_prod)
                used_set = set(current_primes)
                candidates = [q for q in primes if q not in used_set and q <= max_allowed_q]
                if candidates:
                    q_new = random.choice(candidates)
                    current_primes[i] = q_new
        
        current_score = score_from_primes(current_primes)
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            current_prod = product(current_primes)
            used_set = set(current_primes)
            
            for i in range(len(current_primes)):
                if time.time() > deadline - 10.0:
                    break
                p_old = current_primes[i]
                max_allowed_q = int((10**18 * p_old) / current_prod)
                
                for q in primes:
                    if q in used_set:
                        continue
                    if q > max_allowed_q:
                        continue
                    
                    new_primes = current_primes[:]
                    new_primes[i] = q
                    new_score = score_from_primes(new_primes)
                    
                    if new_score > current_score:
                        current_primes = new_primes
                        current_score = new_score
                        improved = True
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_primes = current_primes[:]
                        break
    
    n = product(best_primes)
    if n > 10**18:
        n = product(base_primes)
    return n
