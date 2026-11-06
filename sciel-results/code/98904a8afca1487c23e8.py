"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import defaultdict

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Leave 5s margin for the 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_exhausted():
    return _elapsed() > _TIME_LIMIT

def _score(chosen):
    """Compute the score sum 1/(512 - a) for a set of chosen integers."""
    return sum(1.0 / (512.0 - a) for a in chosen)

def _is_coprime_set(chosen):
    """Check if all pairs in chosen are coprime."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if gcd(chosen[i], chosen[j]) != 1:
                return False
    return True

def _get_primes_below(n):
    """Get all primes below n."""
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n, i):
                sieve[j] = False
    return [i for i in range(2, n) if sieve[i]]

def _factorize(n, primes):
    """Return the set of prime factors of n."""
    factors = set()
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.add(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.add(temp)
    return factors

def _greedy_init_matching():
    """
    Maximum-weight matching inspired greedy:
    - Left partition: primes below 512
    - Right partition: integers 1..511
    - Edge weight for (p, a): 1/(512-a) if p|a
    - Greedily select integer with highest weight-to-conflict ratio
    """
    primes = _get_primes_below(512)
    
    # Precompute prime factors for each integer
    factor_map = {}
    for a in range(1, 512):
        factor_map[a] = _factorize(a, primes)
    
    # Precompute weights
    weights = {a: 1.0 / (512.0 - a) for a in range(1, 512)}
    
    chosen = []
    chosen_set = set()
    # Track which primes are "used" (appear in chosen numbers)
    used_primes = set()
    
    for a in range(511, 0, -1):
        if a in chosen_set:
            continue
        
        # Check coprimality with all chosen
        if any(gcd(a, b) != 1 for b in chosen):
            continue
        
        # Calculate conflict score: how many new prime conflicts this creates
        a_factors = factor_map[a]
        new_conflicts = len(a_factors - used_primes)
        
        # Weight-to-conflict ratio (higher is better)
        # If no new conflicts, ratio is infinite (very good)
        if new_conflicts == 0:
            ratio = float('inf')
        else:
            ratio = weights[a] / new_conflicts
        
        # For now, just add if coprime (we'll use ratio for ordering later)
        chosen.append(a)
        chosen_set.add(a)
        used_primes |= a_factors
    
    return chosen

def _greedy_init_weighted():
    """
    Weighted greedy: iteratively select the integer that maximizes
    marginal benefit / conflict cost.
    """
    primes = _get_primes_below(512)
    
    # Precompute prime factors for each integer
    factor_map = {}
    for a in range(1, 512):
        factor_map[a] = _factorize(a, primes)
    
    # Precompute weights
    weights = {a: 1.0 / (512.0 - a) for a in range(1, 512)}
    
    chosen = []
    chosen_set = set()
    used_primes = set()
    
    remaining = set(range(1, 512))
    
    while remaining and not _time_exhausted():
        best_a = None
        best_ratio = -1
        
        for a in remaining:
            # Check coprimality with all chosen
            if any(gcd(a, b) != 1 for b in chosen):
                continue
            
            a_factors = factor_map[a]
            new_conflicts = len(a_factors - used_primes)
            
            if new_conflicts == 0:
                ratio = weights[a] * 1000  # Boost for no new conflicts
            else:
                ratio = weights[a] / new_conflicts
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_a = a
        
        if best_a is None:
            break
        
        chosen.append(best_a)
        chosen_set.add(best_a)
        remaining.remove(best_a)
        used_primes |= factor_map[best_a]
    
    return chosen

def _local_search(chosen):
    """Swap-based local search to improve the score."""
    best = list(chosen)
    best_score = _score(best)
    
    while not _time_exhausted():
        improved = False
        
        # Try adding elements first (highest value candidates)
        for a in range(511, 0, -1):
            if _time_exhausted():
                break
            if a in best:
                continue
            if not all(gcd(a, b) == 1 for b in best):
                continue
            
            new_set = best + [a]
            new_score = _score(new_set)
            
            if new_score > best_score + 1e-15:
                best = new_set
                best_score = new_score
                improved = True
                break
        
        # Try swaps: remove one, add another
        if not improved:
            for i in range(len(best)):
                if _time_exhausted():
                    break
                removed = best[i]
                candidate_set = best[:i] + best[i+1:]
                
                for a in range(511, 0, -1):
                    if a in candidate_set:
                        continue
                    if not all(gcd(a, b) == 1 for b in candidate_set):
                        continue
                    
                    new_set = candidate_set + [a]
                    new_score = _score(new_set)
                    
                    if new_score > best_score + 1e-15:
                        best = new_set
                        best_score = new_score
                        improved = True
                        break
                
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    # Try multiple initialization strategies and keep the best
    candidates = []
    
    # Strategy 1: Simple greedy (largest first)
    chosen1 = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen1):
            chosen1.append(a)
    candidates.append(chosen1)
    
    # Strategy 2: Weighted greedy (matching-inspired)
    if not _time_exhausted():
        chosen2 = _greedy_init_weighted()
        candidates.append(chosen2)
    
    # Strategy 3: Randomized restarts with local search
    import random
    rng = random.Random(42)
    
    for restart in range(20):
        if _time_exhausted():
            break
        
        # Randomized greedy initialization
        chosen_r = []
        available = list(range(1, 512))
        rng.shuffle(available)
        
        # Sort by weight descending with some randomness
        for a in sorted(available, key=lambda x: (-1.0/(512.0-x), rng.random())):
            if any(gcd(a, b) != 1 for b in chosen_r):
                continue
            chosen_r.append(a)
            if len(chosen_r) > 30:  # Limit size for speed
                break
        
        candidates.append(chosen_r)
    
    # Apply local search to each candidate and keep the best
    best_overall = None
    best_score_overall = -1
    
    for cand in candidates:
        if _time_exhausted():
            break
        
        improved = _local_search(cand)
        s = _score(improved)
        
        if s > best_score_overall:
            best_score_overall = s
            best_overall = improved
    
    return best_overall if best_overall is not None else candidates[0]
# EVOLVE-BLOCK-END
