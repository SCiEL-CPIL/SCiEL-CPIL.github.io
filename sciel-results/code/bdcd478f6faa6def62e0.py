"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from math import log, gcd, log2

# EVOLVE-BLOCK-START

def _score_from_exponents(exponents, primes):
    """
    Given exponents (list of ints) for each prime in primes (list of ints),
    compute n = prod p_i^e_i and the score tau_perp(n).
    Score = number of adjacent divisor pairs (in increasing order) with gcd 1.
    """
    n = 1
    for e, p in zip(exponents, primes):
        if e > 0:
            n *= p ** e
    if n < 2 or n > 10**18:
        return 0, n
    
    # Generate all divisors
    divisors = [1]
    for e, p in zip(exponents, primes):
        if e == 0:
            continue
        new_divs = []
        for d in divisors:
            pk = 1
            for _ in range(e):
                pk *= p
                new_divs.append(d * pk)
        divisors.extend(new_divs)
    
    divisors.sort()
    
    score = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i+1]) == 1:
            score += 1
    
    return score, n


def _prime_sieve(limit):
    """Generate all primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    return [i for i in range(2, limit+1) if sieve[i]]


def _greedy_polish(exponents, primes, deadline):
    """
    Local search: try to add or remove 1 from exponents to improve score.
    """
    current_score, current_n = _score_from_exponents(exponents, primes)
    if current_n < 2 or current_n > 10**18:
        return exponents, current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(primes)):
            if time.time() >= deadline:
                break
            # Try increasing exponent
            if exponents[i] + 1 >= 0:
                new_exp = exponents[:]
                new_exp[i] += 1
                s, n = _score_from_exponents(new_exp, primes)
                if n <= 10**18 and s > current_score:
                    exponents = new_exp
                    current_score = s
                    improved = True
            # Try decreasing exponent
            if exponents[i] > 0:
                new_exp = exponents[:]
                new_exp[i] -= 1
                s, n = _score_from_exponents(new_exp, primes)
                if n <= 10**18 and s > current_score:
                    exponents = new_exp
                    current_score = s
                    improved = True
            # Try setting to 0
            if exponents[i] > 0:
                new_exp = exponents[:]
                new_exp[i] = 0
                s, n = _score_from_exponents(new_exp, primes)
                if n <= 10**18 and s > current_score:
                    exponents = new_exp
                    current_score = s
                    improved = True
    return exponents, current_score


def _simulated_annealing(primes, deadline, seed=None):
    """
    Simulated annealing over exponent vectors.
    """
    if seed is not None:
        random.seed(seed)
    
    n_primes = len(primes)
    
    # Start with all exponents = 1
    exponents = [1] * n_primes
    current_score, current_n = _score_from_exponents(exponents, primes)
    
    if current_n > 10**18:
        # Scale down
        while current_n > 10**18:
            i = random.randint(0, n_primes-1)
            if exponents[i] > 0:
                exponents[i] -= 1
                current_score, current_n = _score_from_exponents(exponents, primes)
    
    best_exponents = exponents[:]
    best_score = current_score
    
    # Temperature schedule
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            break
        
        # Random move: pick a prime and adjust exponent by -1, 0, or +1
        i = random.randint(0, n_primes - 1)
        delta = random.choice([-1, 1, 1, -1])
        new_exp = exponents[:]
        new_exp[i] = max(0, new_exp[i] + delta)
        
        new_score, new_n = _score_from_exponents(new_exp, primes)
        
        if new_n > 10**18 or new_n < 2:
            continue
        
        delta_score = new_score - current_score
        
        if delta_score >= 0 or random.random() < math.exp(delta_score / T):
            exponents = new_exp
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_exponents = exponents[:]
    
    return best_exponents, best_score


def _local_search_random(primes, deadline, seed=None):
    """
    Random restart local search.
    """
    if seed is not None:
        random.seed(seed)
    
    n_primes = len(primes)
    best_exponents = None
    best_score = 0
    
    while time.time() < deadline:
        # Random initial exponents
        exponents = [random.randint(0, 3) for _ in range(n_primes)]
        # Ensure at least one prime is used
        if all(e == 0 for e in exponents):
            exponents[random.randint(0, n_primes-1)] = 1
        
        score, n = _score_from_exponents(exponents, primes)
        if n < 2 or n > 10**18:
            continue
        
        # Polish
        exponents, score = _greedy_polish(exponents, primes, deadline)
        
        if score > best_score:
            best_score = score
            best_exponents = exponents[:]
    
    return best_exponents, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate primes up to 2000
    primes = _prime_sieve(2000)
    
    best_score = 0
    best_exponents = None
    
    # Strategy 1: Simulated annealing with multiple restarts
    num_sa_runs = 5
    sa_time_each = (deadline - start_time) / (num_sa_runs + 3)
    
    for run in range(num_sa_runs):
        if time.time() >= deadline:
            break
        sa_deadline = min(deadline, time.time() + sa_time_each)
        exponents, score = _simulated_annealing(primes, sa_deadline, seed=42 + run)
        if score > best_score:
            best_score = score
            best_exponents = exponents[:]
    
    # Strategy 2: Local search with random restarts
    if time.time() < deadline - 5:
        ls_deadline = min(deadline - 2, time.time() + (deadline - time.time()) / 2)
        exponents, score = _local_search_random(primes, ls_deadline, seed=123)
        if score > best_score:
            best_score = score
            best_exponents = exponents[:]
    
    # Final polish
    if best_exponents is not None and time.time() < deadline - 1:
        best_exponents, best_score = _greedy_polish(best_exponents, primes, deadline)
    
    # Compute final n
    if best_exponents is None:
        # Fallback: product of first 8 primes
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    n = 1
    for e, p in zip(best_exponents, primes):
        if e > 0:
            n *= p ** e
    
    if n < 2 or n > 10**18:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    return n

# EVOLVE-BLOCK-END
