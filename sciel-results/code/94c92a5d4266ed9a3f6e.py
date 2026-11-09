import math
import time
import random
from itertools import combinations

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number (product of given primes), sorted."""
    n = 1
    for p in primes:
        n *= p
    # Generate all 2^k divisors
    k = len(primes)
    divs = [1]
    for i, p in enumerate(primes):
        divs += [d * p for d in divs[:len(divs)//2]]
    divs.sort()
    return divs, n

def _tau_perp(divs):
    """Count adjacent coprime pairs in sorted divisor list."""
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _evaluate(primes):
    """Evaluate tau_perp for squarefree n = product of primes."""
    n = 1
    for p in primes:
        n *= p
    if n > 10**18:
        return -1
    divs, _ = _divisors_of_squarefree(primes)
    return _tau_perp(divs)

def _greedy_polish(primes, all_primes, deadline):
    """Try to improve by swapping primes in/out."""
    current = set(primes)
    current_score = _evaluate(list(current))
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Try removing a prime
        for p in list(current):
            if time.time() > deadline:
                break
            trial = current - {p}
            if not trial:
                continue
            s = _evaluate(list(trial))
            if s > current_score:
                current = trial
                current_score = s
                improved = True
                break
        # Try adding a prime
        for p in all_primes:
            if time.time() > deadline:
                break
            if p in current:
                continue
            trial = current | {p}
            s = _evaluate(list(trial))
            if s > current_score:
                current = trial
                current_score = s
                improved = True
                break
    return list(current), current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate primes up to ~2000
    limit = 2000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    all_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    # Initial solution: use smallest primes that fit
    best_primes = []
    prod = 1
    for p in all_primes:
        if prod * p <= 10**18:
            best_primes.append(p)
            prod *= p
        else:
            break
    best_score = _evaluate(best_primes)
    
    if time.time() > deadline:
        return 1
    for p in best_primes:
            pass
    n_best = 1
    for p in best_primes:
        n_best *= p
    
    # Simulated annealing
    current = list(best_primes)
    current_score = best_score
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # Generate a move: swap one prime in/out
        if len(current) > 1 and random.random() < 0.5:
            # Remove a random prime
            idx = random.randint(0, len(current) - 1)
            trial = current[:idx] + current[idx+1:]
        else:
            # Add a random prime not in current
            candidates = [p for p in all_primes if p not in current]
            if not candidates:
                break
            idx = random.randint(0, len(candidates) - 1)
            trial = current + [candidates[idx]]
        
        trial_score = _evaluate(trial)
        
        if trial_score > current_score:
            current = trial
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best_primes = list(current)
        elif trial_score >= 0:
            # Accept worse with probability
            delta = current_score - trial_score
            if random.random() < math.exp(-delta / max(T, T_min)):
                current = trial
                current_score = trial_score
        
        T = max(T * cooling, T_min)
    
    # Final greedy polish
    if time.time() < deadline - 2.0:
        polished, polish_score = _greedy_polish(best_primes, all_primes, deadline)
        if polish_score > best_score:
            best_primes = polished
            best_score = polish_score
    
    # Compute final n
    n = 1
    for p in best_primes:
        n *= p
    
    if n < 2 or n > 10**18:
        return 2
    
    return n
