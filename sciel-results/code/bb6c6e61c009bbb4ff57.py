"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from math import gcd

def _factorize(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _score(chosen):
    """Compute the score: sum of 1/(512 - a) for all chosen a."""
    return sum(1.0 / (512 - a) for a in chosen)

def _get_all_primes_below(n):
    """Get all primes below n."""
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n, i):
                sieve[j] = False
    return [i for i in range(n) if sieve[i]]

def _hungarian_optimal(deadline):
    """
    Use Hungarian algorithm to find optimal subset.
    Model: each candidate a has weight w(a) = 1/(512-a).
    Constraint: pairwise coprime means no shared prime factors.
    
    We can model this as: assign each prime to at most one number.
    But this is complex. Instead, use a greedy + local search with better initialization.
    """
    # Precompute factorizations
    factorizations = {}
    for a in range(1, 512):
        factorizations[a] = _factorize(a)
    
    # Get all primes up to 511
    all_primes = _get_all_primes_below(512)
    
    # Strategy: For each prime, decide which number "owns" it.
    # A number is valid if all its prime factors are owned by it.
    # We want to maximize sum of weights of valid numbers.
    
    # This is still complex. Let's use a different approach:
    # Greedy from high to low, but with lookahead.
    
    chosen = []
    chosen_factors = []
    chosen_set = set()
    
    # Sort candidates by weight descending (i.e., by a descending since 1/(512-a) increases with a)
    candidates = list(range(511, 0, -1))
    
    for a in candidates:
        if time.time() > deadline:
            break
        if a in chosen_set:
            continue
        af = factorizations[a]
        # Check if coprime to all chosen
        compatible = True
        for cf in chosen_factors:
            if af & cf:
                compatible = False
                break
        if compatible:
            chosen.append(a)
            chosen_factors.append(af)
            chosen_set.add(a)
    
    return chosen

def _local_search_improved(chosen, deadline, factorizations):
    """Improved local search with prime-factor based conflict detection."""
    best = list(chosen)
    best_score = _score(best)
    best_factors = [factorizations[a] for a in best]
    chosen_set = set(best)
    
    # Precompute all prime factors that are in use
    used_primes = set()
    for f in best_factors:
        used_primes.update(f)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-for-1 swaps
        for i in range(len(best)):
            if time.time() > deadline:
                break
            removed = best[i]
            removed_factors = best_factors[i]
            
            # Find best replacement
            best_replacement = None
            best_gain = 0.0
            
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                af = factorizations[a]
                # Check if a is coprime to all other chosen elements
                compatible = True
                for j in range(len(best)):
                    if j == i:
                        continue
                    if af & best_factors[j]:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                gain = 1.0/(512 - a) - 1.0/(512 - removed)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_replacement = a
            
            if best_replacement is not None:
                # Perform the swap
                chosen_set.remove(removed)
                chosen_set.add(best_replacement)
                best[i] = best_replacement
                best_factors[i] = factorizations[best_replacement]
                best_score += best_gain
                improved = True
    
    return best, best_score

def _try_additions(chosen, deadline, factorizations):
    """Try to add more elements if possible."""
    best = list(chosen)
    best_score = _score(best)
    best_factors = [factorizations[a] for a in best]
    chosen_set = set(best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in range(511, 0, -1):
            if time.time() > deadline:
                break
            if a in chosen_set:
                continue
            af = factorizations[a]
            compatible = True
            for cf in best_factors:
                if af & cf:
                    compatible = False
                    break
            if compatible:
                best.append(a)
                best_factors.append(af)
                chosen_set.add(a)
                best_score += 1.0/(512 - a)
                improved = True
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Precompute factorizations
    factorizations = {}
    for a in range(1, 512):
        factorizations[a] = _factorize(a)
    
    # Phase 1: Greedy initialization
    chosen = _hungarian_optimal(deadline)
    if not chosen:
        return [511]
    
    # Phase 2: Try to add more elements
    if time.time() < deadline:
        chosen, _ = _try_additions(chosen, deadline, factorizations)
    
    # Phase 3: Local search
    if time.time() < deadline:
        chosen, best_score = _local_search_improved(chosen, deadline, factorizations)
    
    return chosen
# EVOLVE-BLOCK-END
