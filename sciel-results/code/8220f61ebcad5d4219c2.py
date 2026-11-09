"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _score(n):
    """Compute tau_perp(n): number of adjacent coprime divisor pairs."""
    if n < 2:
        return 0
    # Get all divisors
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    count = 0
    gcd = math.gcd
    prev = divs[0]
    for curr in divs[1:]:
        if gcd(prev, curr) == 1:
            count += 1
        prev = curr
    return count

def _score_from_primes(primes):
    """Compute score given a set of primes (n = product of primes)."""
    n = 1
    for p in primes:
        n *= p
    if n > 10**18:
        return -1
    
    # Generate divisors directly from primes (square-free assumption)
    # This is O(2^k) instead of O(sqrt(n))
    divs = [1]
    for p in primes:
        # Create new divisors by multiplying existing ones by p
        # Use list comprehension for speed
        new_divs = [d * p for d in divs]
        divs.extend(new_divs)
    
    divs.sort()
    
    count = 0
    gcd = math.gcd
    prev = divs[0]
    for curr in divs[1:]:
        if gcd(prev, curr) == 1:
            count += 1
        prev = curr
        
    return count

def _generate_divisors(primes):
    """Generate all divisors from prime list, sorted."""
    divs = [1]
    for p in primes:
        new_divs = []
        for d in divs:
            new_divs.append(d * p)
        divs.extend(new_divs)
    divs.sort()
    return divs

def _score_divisors(divs):
    """Score from sorted divisor list."""
    count = 0
    for j in range(len(divs) - 1):
        if math.gcd(divs[j], divs[j+1]) == 1:
            count += 1
    return count

def _prime_list(limit):
    """Generate primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    return [i for i in range(2, limit+1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Generate primes up to 300
    primes_pool = _prime_list(300)
    n_primes = len(primes_pool)
    
    # Initial solution: first few primes
    # Find max k such that product of first k primes <= 10^18
    max_k = 0
    prod = 1
    for i, p in enumerate(primes_pool):
        if prod * p > 10**18:
            break
        prod *= p
        max_k = i + 1
        
    best_set = set(primes_pool[:max_k])
    best_score = _score_from_primes(list(best_set))
    
    if best_score < 0:
        best_set = set(primes_pool[:max(1, max_k-1)])
        best_score = _score_from_primes(list(best_set))
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    max_iter = 2000  # Reduced drastically to avoid timeout
    
    current_set = set(best_set)
    current_score = best_score
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        # Additional safety: break if too slow per iteration
        if it > 0 and it % 100 == 0:
            if time.time() - start_time > 100: # Hard stop at 100s
                break
        
        # Random move: swap one prime in/out
        if random.random() < 0.5:
            # Try to add a prime
            candidates = [p for p in primes_pool if p not in current_set]
            if not candidates:
                continue
            new_p = random.choice(candidates)
            new_set = current_set | {new_p}
            # Check if product exceeds 10^18
            prod = 1
            for p in new_set:
                prod *= p
                if prod > 10**18:
                    break
            if prod > 10**18:
                continue
            new_score = _score_from_primes(list(new_set))
            if new_score < 0:
                continue
        else:
            # Try to remove a prime
            if len(current_set) <= 2:
                continue
            to_remove = random.choice(list(current_set))
            new_set = current_set - {to_remove}
            prod = 1
            for p in new_set:
                prod *= p
                if prod > 10**18:
                    break
            if prod > 10**18:
                continue
            new_score = _score_from_primes(list(new_set))
            if new_score < 0:
                continue
        
        delta = new_score - current_score
        if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Try local improvement: try swapping primes
    # Limit this phase strictly
    local_start = time.time()
    for _ in range(50):
        if time.time() - local_start > 5.0:
            break
        improved = False
        cur_list = list(best_set)
        for i in range(len(cur_list)):
            for p in primes_pool:
                if p in best_set:
                    continue
                new_set = (best_set - {cur_list[i]}) | {p}
                prod = 1
                for q in new_set:
                    prod *= q
                    if prod > 10**18:
                        break
                if prod > 10**18:
                    continue
                s = _score_from_primes(list(new_set))
                if s > best_score:
                    best_set = new_set
                    best_score = s
                    improved = True
        if not improved:
            break
    
    result = 1
    for p in best_set:
        result *= p
    
    if result < 2:
        result = 2
    if result > 10**18:
        result = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    return result

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
