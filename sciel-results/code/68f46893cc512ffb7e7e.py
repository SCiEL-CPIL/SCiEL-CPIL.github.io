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

def tau_perp(n):
    # n is the product of primes in the list
    # We need the list of primes to generate divisors efficiently
    # But construct() returns n. 
    # Wait, the interface is construct() -> n.
    # But to calculate the score during search, we have the primes.
    # The evaluator calls construct() and then calculates the score on the returned n.
    # So my local search needs to calculate the score on the candidate n.
    # I will keep the primes list in the search.
    pass

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
    deadline = start_time + 145.0
    
    # Generate primes up to 200
    primes = sieve(200)
    
    # Initial set: 15 smallest primes
    current_primes = primes[:15]
    current_score = score_from_primes(current_primes)
    best_score = current_score
    best_primes = current_primes[:]
    
    # Local Search
    improved = True
    while improved:
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(len(current_primes)):
            if time.time() > deadline - 5.0:
                break
            p_old = current_primes[i]
            # Try replacing with larger primes
            for q in primes:
                if q <= p_old:
                    continue
                new_primes = current_primes[:]
                new_primes[i] = q
                # Sort to keep consistent? Order doesn't matter for product/divisors
                if product(new_primes) > 10**18:
                    break # Since primes are sorted, larger q will only increase product
                new_score = score_from_primes(new_primes)
                if new_score > current_score:
                    current_score = new_score
                    current_primes = new_primes
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_primes = current_primes[:]
                    break # Restart search with new current_primes? 
                         # Or continue? 
                         # Usually better to restart or continue with updated current.
                         # Let's continue with updated current.
    
    # Try some random restarts if time permits
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            # Random set of 15 primes
            # Ensure product < 10^18
            # Pick random primes from a range
            # To ensure product limit, pick from smaller range
            # e.g., primes between 5 and 100
            cand_primes = random.sample(primes[2:50], 15)
            if product(cand_primes) > 10**18:
                continue
            s = score_from_primes(cand_primes)
            if s > best_score:
                best_score = s
                best_primes = cand_primes
    
    n = product(best_primes)
    return n
