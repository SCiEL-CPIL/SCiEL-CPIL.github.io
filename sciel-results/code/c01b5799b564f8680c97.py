import math
import time
import random
from math import gcd

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_up_to(300)

def _tau_perp(primes, exponents):
    """
    Compute tau_perp score from prime factorization.
    """
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
    
    g = gcd
    d = divisors
    n = len(d)
    count = 0
    for i in range(n - 1):
        if g(d[i], d[i + 1]) == 1:
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
    
    best_score = 0
    best_n = 2
    
    # Candidate bases: Primorials of the first k primes
    bases = []
    prod = 1
    for i, p in enumerate(_PRIMES):
        if prod * p > max_n:
            break
        prod *= p
        bases.append((i + 1, prod))
    
    # Use top 4 bases
    search_bases = bases[-4:]
    
    # Build seeds: primorial and 2*primorial variants
    seeds = []
    for k, base_prod in search_bases:
        seeds.append((k, base_prod, 1))
        if base_prod * 2 <= max_n:
            seeds.append((k, base_prod * 2, 2))
    
    rng = random.Random(42)
    
    for k, base_prod, mult in seeds:
        if time.time() > deadline:
            break
            
        # Initial state
        current_primes = _PRIMES[:k]
        current_exps = [1] * k
        
        if mult > 1:
            for p in current_primes:
                if mult % p == 0:
                    current_exps[current_primes.index(p)] += 1
                    break
        
        current_n = base_prod
        current_score = _tau_perp(current_primes, current_exps)
        
        if current_score > best_score:
            best_score = current_score
            best_n = current_n
            
        # Local Search with first-improvement and randomized move order
        improved = True
        iterations = 0
        max_iterations = 5000
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Build list of possible moves
            moves = []
            
            # +1 moves
            for i in range(k):
                if current_n * current_primes[i] <= max_n:
                    moves.append(('inc', i))
            
            # -1 moves
            for i in range(k):
                if current_exps[i] > 1 and current_n // current_primes[i] >= 2:
                    moves.append(('dec', i))
            
            # Swap moves (shift from higher exp to lower)
            for i in range(k):
                if current_exps[i] <= 1:
                    continue
                for j in range(k):
                    if i == j:
                        continue
                    if current_exps[i] > current_exps[j]:
                        new_n = (current_n // current_primes[i]) * current_primes[j]
                        if new_n <= max_n:
                            moves.append(('swap', i, j))
            
            if not moves:
                break
            
            # Shuffle moves for randomization
            rng.shuffle(moves)
            
            # First-improvement: try moves until we find one that improves
            for move in moves:
                if time.time() > deadline:
                    break
                
                if move[0] == 'inc':
                    i = move[1]
                    new_exps = current_exps[:]
                    new_exps[i] += 1
                    new_n = current_n * current_primes[i]
                elif move[0] == 'dec':
                    i = move[1]
                    new_exps = current_exps[:]
                    new_exps[i] -= 1
                    new_n = current_n // current_primes[i]
                else:  # swap
                    i, j = move[1], move[2]
                    new_exps = current_exps[:]
                    new_exps[i] -= 1
                    new_exps[j] += 1
                    new_n = (current_n // current_primes[i]) * current_primes[j]
                
                score = _tau_perp(current_primes, new_exps)
                if score > current_score:
                    current_exps = new_exps
                    current_n = new_n
                    current_score = score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
                    break  # First improvement, restart iteration
    
    return best_n
