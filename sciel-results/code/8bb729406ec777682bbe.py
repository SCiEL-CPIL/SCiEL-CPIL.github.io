"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

# Precompute prime factorization for all numbers 1..511
_N = 512
_primes_under_512 = []
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(2, n+1) if is_prime[i]]

_primes_under_512 = _sieve(_N - 1)
_prime_set = set(_primes_under_512)

# Factorization: for each number 1..511, store set of prime factors
_factors = [None] * (_N)
for i in range(1, _N):
    n = i
    pf = set()
    for p in _primes_under_512:
        if p * p > n:
            break
        if n % p == 0:
            pf.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        pf.add(n)
    _factors[i] = pf

# Precompute score values: 1/(512 - a) for a in 1..511
_score_val = [0.0] * _N
_score_frac = [None] * _N
for a in range(1, _N):
    _score_val[a] = 1.0 / (_N - a)
    _score_frac[a] = Fraction(1, _N - a)

# Precompute pairwise coprime check using bitmasks for small primes
# Actually, let's use the factor sets directly for speed

def _is_coprime(a, b):
    """Check if a and b are coprime using precomputed factors."""
    fa = _factors[a]
    fb = _factors[b]
    if not fa or not fb:
        return True
    # Check intersection
    if len(fa) > len(fb):
        fa, fb = fb, fa
    for p in fa:
        if p in fb:
            return False
    return True

def _greedy_construct(order=None):
    """
    Greedily construct a set of pairwise coprime integers.
    If order is None, use descending order by score (i.e., a from 511 down to 1).
    If order is a list, use that order.
    Returns (list_of_chosen, score_float, score_frac).
    """
    if order is None:
        order = list(range(_N - 1, 0, -1))
    
    chosen = []
    chosen_factors = set()
    score_float = 0.0
    score_frac = Fraction(0, 1)
    
    for a in order:
        fa = _factors[a]
        # Check if a shares any factor with already chosen
        if not fa:
            # a=1, always coprime
            chosen.append(a)
            score_float += _score_val[a]
            score_frac += _score_frac[a]
        else:
            conflict = False
            for p in fa:
                if p in chosen_factors:
                    conflict = True
                    break
            if not conflict:
                chosen.append(a)
                chosen_factors.update(fa)
                score_float += _score_val[a]
                score_frac += _score_frac[a]
    
    return chosen, score_float, score_frac

def _local_search(chosen, max_iters=100, deadline=None):
    """
    Try to improve the set by swapping elements.
    For each element in chosen, try replacing it with a better element not in chosen.
    Also try adding elements if possible (but we can't add if all are coprime already... 
    actually we can't add because the greedy already tried all in order).
    
    Better approach: try to swap a low-value element with a high-value element that is coprime
    to all other chosen elements.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    chosen_set = set(chosen)
    chosen_factors = set()
    for a in chosen:
        chosen_factors.update(_factors[a])
    
    current_score_float = sum(_score_val[a] for a in chosen)
    
    # Sort chosen by score ascending (lowest first) - these are candidates for removal
    chosen_sorted = sorted(chosen, key=lambda x: _score_val[x])
    
    # Candidate replacements: numbers not in chosen, sorted by score descending
    candidates = [a for a in range(_N - 1, 0, -1) if a not in chosen_set]
    
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        # Try removing the lowest-score element and adding the best candidate
        if not chosen_sorted:
            break
        
        # For each element in chosen (lowest score first), try to remove it
        # and add the best candidate that becomes coprime
        for i, remove_a in enumerate(chosen_sorted):
            if time.time() > deadline:
                break
            
            # Compute factors without remove_a
            new_factors = set()
            for a in chosen:
                if a != remove_a:
                    new_factors.update(_factors[a])
            
            # Find best candidate that is coprime to new_factors
            best_candidate = None
            best_gain = 0.0
            remove_score = _score_val[remove_a]
            
            for cand in candidates:
                if cand in chosen_set:
                    continue
                fc = _factors[cand]
                if fc and any(p in new_factors for p in fc):
                    continue
                gain = _score_val[cand] - remove_score
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
            
            if best_candidate is not None and best_gain > 1e-15:
                # Perform the swap
                chosen.remove(remove_a)
                chosen_set.remove(remove_a)
                chosen.append(best_candidate)
                chosen_set.add(best_candidate)
                
                # Update factors
                new_factors = set()
                for a in chosen:
                    new_factors.update(_factors[a])
                chosen_factors = new_factors
                
                # Re-sort chosen_sorted
                chosen_sorted = sorted(chosen, key=lambda x: _score_val[x])
                current_score_float += best_gain
                improved = True
                break  # Restart the loop after a swap
    
    # Final score
    final_score_float = sum(_score_val[a] for a in chosen)
    final_score_frac = sum((_score_frac[a] for a in chosen), Fraction(0, 1))
    
    return chosen, final_score_float, final_score_frac

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 150s limit, keep 10s buffer
    
    # Phase 1: Greedy construction (very fast)
    greedy_chosen, greedy_score_float, greedy_score_frac = _greedy_construct()
    
    best_chosen = greedy_chosen[:]
    best_score_float = greedy_score_float
    best_score_frac = greedy_score_frac
    
    # Phase 2: Local search on the greedy result
    if time.time() < deadline - 5:
        ls_chosen, ls_score_float, ls_score_frac = _local_search(
            greedy_chosen[:], max_iters=50, deadline=deadline - 2
        )
        if ls_score_float > best_score_float + 1e-15:
            best_chosen = ls_chosen
            best_score_float = ls_score_float
            best_score_frac = ls_score_frac
    
    # Phase 3: Randomized restarts with different orderings
    # Use random perturbations of the order to find better solutions
    rng = random.Random(42)
    
    restart_count = 0
    while time.time() < deadline - 2:
        restart_count += 1
        
        # Create a perturbed order
        # Strategy: mostly descending by score, but with some randomization
        # for the lower-score elements
        base_order = list(range(_N - 1, 0, -1))
        
        # Randomly shuffle a portion of the order
        # Keep the top ~50 elements in order, shuffle the rest
        if restart_count % 3 == 0:
            # Full random shuffle (but biased toward high scores)
            weights = [_score_val[a] ** 2 for a in base_order]
            # Use weighted sampling without replacement
            shuffled = base_order[:]
            for i in range(len(shuffled) - 1, 0, -1):
                j = rng.randint(0, i)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            order = shuffled
        elif restart_count % 3 == 1:
            # Partial shuffle: keep top 30 in order, shuffle the rest
            top_n = 30
            top = base_order[:top_n]
            rest = base_order[top_n:]
            rng.shuffle(rest)
            order = top + rest
        else:
            # Another strategy: sort by score with some noise
            noisy_order = []
            for a in base_order:
                noisy_score = _score_val[a] * rng.uniform(0.8, 1.2)
                noisy_order.append((noisy_score, a))
            noisy_order.sort(key=lambda x: -x[0])
            order = [a for _, a in noisy_order]
        
        # Greedy construct with this order
        restart_chosen, restart_score_float, restart_score_frac = _greedy_construct(order)
        
        if restart_score_float > best_score_float + 1e-15:
            best_chosen = restart_chosen
            best_score_float = restart_score_float
            best_score_frac = restart_score_frac
            
            # Try local search on this improved solution
            if time.time() < deadline - 5:
                ls_chosen, ls_score_float, ls_score_frac = _local_search(
                    best_chosen[:], max_iters=30, deadline=deadline - 2
                )
                if ls_score_float > best_score_float + 1e-15:
                    best_chosen = ls_chosen
                    best_score_float = ls_score_float
                    best_score_frac = ls_score_frac
    
    # Phase 4: Final polish - try all single swaps from the best solution
    if time.time() < deadline - 1:
        chosen_set = set(best_chosen)
        best_chosen, _, _ = _local_search(
            best_chosen[:], max_iters=20, deadline=deadline - 0.5
        )
        best_score_float = sum(_score_val[a] for a in best_chosen)
        best_score_frac = sum((_score_frac[a] for a in best_chosen), Fraction(0, 1))
    
    return best_chosen
# EVOLVE-BLOCK-END
