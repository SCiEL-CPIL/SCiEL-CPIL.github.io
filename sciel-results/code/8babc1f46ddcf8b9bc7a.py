import math
import time

def _sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _get_divisors(prime_set):
    """Get all divisors of a squarefree number given its prime factors."""
    divs = [1]
    for p in prime_set:
        new_divs = []
        for d in divs:
            new_divs.append(d)
            new_divs.append(d * p)
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(n, prime_set):
    """Count adjacent coprime pairs in sorted divisors of squarefree n."""
    divs = _get_divisors(prime_set)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Find all primes up to a reasonable limit.
    # Product of first 15 primes ≈ 6.15e17, first 16 ≈ 3.25e19 > 1e18.
    # So we need primes up to around 53 at minimum, but let's go a bit higher
    # to allow swaps that might improve the score.
    max_prime_limit = 200
    all_primes = _sieve(max_prime_limit)
    
    # Determine how many of the smallest primes we can fit.
    # Start with the first k primes where product <= 1e18.
    prod = 1
    max_k = 0
    for i, p in enumerate(all_primes):
        if prod * p <= 10**18:
            prod *= p
            max_k = i + 1
        else:
            break
    
    # Initial solution: first max_k primes
    initial_set = all_primes[:max_k]
    best_score = _tau_perp(prod, initial_set)
    best_n = prod
    best_set = list(initial_set)
    
    # Local search: try swapping one prime out and one prime in
    # to improve the score while keeping n <= 1e18.
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(len(best_set)):
            if time.time() > deadline - 5.0:
                break
            for j in range(len(all_primes)):
                if time.time() > deadline - 5.0:
                    break
                # Skip if same prime or already in set
                if all_primes[j] == best_set[i]:
                    continue
                if all_primes[j] in best_set:
                    continue
                
                # Try swap
                new_set = list(best_set)
                new_set[i] = all_primes[j]
                
                # Check product
                new_prod = 1
                for p in new_set:
                    new_prod *= p
                    if new_prod > 10**18:
                        break
                if new_prod > 10**18:
                    continue
                
                new_score = _tau_perp(new_prod, new_set)
                if new_score > best_score:
                    best_score = new_score
                    best_n = new_prod
                    best_set = new_set
                    improved = True
    
    # Also try adding a prime if possible (increase k)
    if time.time() < deadline - 5.0:
        for j in range(len(all_primes)):
            if time.time() > deadline - 5.0:
                break
            if all_primes[j] in best_set:
                continue
            new_set = list(best_set) + [all_primes[j]]
            new_prod = best_n * all_primes[j]
            if new_prod > 10**18:
                continue
            new_score = _tau_perp(new_prod, new_set)
            if new_score > best_score:
                best_score = new_score
                best_n = new_prod
                best_set = new_set
    
    # Try removing a prime if it improves score (unlikely but check)
    if time.time() < deadline - 5.0:
        for i in range(len(best_set)):
            if time.time() > deadline - 5.0:
                break
            new_set = list(best_set)
            new_set.pop(i)
            new_prod = 1
            for p in new_set:
                new_prod *= p
            new_score = _tau_perp(new_prod, new_set)
            if new_score > best_score:
                best_score = new_score
                best_n = new_prod
                best_set = new_set
    
    return best_n
