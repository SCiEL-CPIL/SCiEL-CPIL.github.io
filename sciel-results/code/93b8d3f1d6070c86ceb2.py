"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
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

# Precompute prime factors for all numbers 1..511
_FACTORS = [set()] + [_factorize(i) for i in range(1, 512)]

def _FACTORIZE_CHECK(n):
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True

# Build prime-to-bit mapping
_PRIMES = [p for p in range(2, 512) if _FACTORIZE_CHECK(p)]
_PRIME_TO_BIT = {p: i for i, p in enumerate(_PRIMES)}
_NUM_PRIMES = len(_PRIMES)

# Build bitmask for each number
def _to_bitmask(factor_set):
    mask = 0
    for p in factor_set:
        if p in _PRIME_TO_BIT:
            mask |= (1 << _PRIME_TO_BIT[p])
    return mask

_MASKS = [0] * 512
for i in range(1, 512):
    _MASKS[i] = _to_bitmask(_FACTORS[i])

# Precompute reciprocal values for fast scoring
_RECIP = [0.0] * 512
for i in range(1, 512):
    _RECIP[i] = 1.0 / (512 - i)

def _union_mask(nums):
    m = 0
    for a in nums:
        m |= _MASKS[a]
    return m

def _is_coprime_mask(a, union_m):
    """Check if a is coprime to all numbers represented by union_m."""
    return (_MASKS[a] & union_m) == 0

def _score(chosen):
    """Compute the score: sum of 1/(512 - a) for all chosen a."""
    return sum(_RECIP[a] for a in chosen)

def _greedy_from_high(deadline=None):
    """Greedy: pick from high to low, adding if coprime to all previously chosen."""
    chosen = []
    union_m = 0
    for a in range(511, 0, -1):
        if deadline is not None and time.time() > deadline:
            break
        if _is_coprime_mask(a, union_m):
            chosen.append(a)
            union_m |= _MASKS[a]
    return chosen

def _local_search(chosen, deadline):
    """Local search with swap and add moves using bitmask checks."""
    best = list(chosen)
    best_score = _score(best)
    chosen_set = set(best)
    union_m = _union_mask(best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try swap moves: replace element i with candidate a
        for i in range(len(best)):
            if time.time() > deadline:
                break
            removed = best[i]
            # Union mask without element i
            rest_m = union_m & ~_MASKS[removed]
            
            # Try adding each candidate not in best, in order of decreasing value
            for a in range(511, 0, -1):
                if time.time() > deadline:
                    break
                if a in chosen_set:
                    continue
                if _is_coprime_mask(a, rest_m):
                    new_score = best_score - _RECIP[removed] + _RECIP[a]
                    if new_score > best_score + 1e-12:
                        # Perform swap
                        chosen_set.remove(removed)
                        chosen_set.add(a)
                        best[i] = a
                        union_m = rest_m | _MASKS[a]
                        best_score = new_score
                        improved = True
                        break
        
        # Try add moves: add a new element if it improves score
        for a in range(511, 0, -1):
            if time.time() > deadline:
                break
            if a in chosen_set:
                continue
            if _is_coprime_mask(a, union_m):
                # Adding always improves since 1/(512-a) > 0
                chosen_set.add(a)
                best.append(a)
                union_m |= _MASKS[a]
                best_score += _RECIP[a]
                improved = True
    
    return best, best_score

def _simulated_annealing(chosen, deadline):
    """Simulated annealing to escape local optima."""
    best = list(chosen)
    best_score = _score(best)
    current = list(chosen)
    current_score = _score(current)
    chosen_set = set(current)
    union_m = _union_mask(current)
    
    t = 1.0
    cooling = 0.995
    
    while time.time() < deadline:
        t *= cooling
        if t < 0.001:
            break
        
        # Try a random swap
        if len(current) > 1:
            i = random.randint(0, len(current) - 1)
            removed = current[i]
            rest_m = union_m & ~_MASKS[removed]
            
            # Find a random candidate that's coprime to rest
            candidates = [a for a in range(511, 0, -1) 
                        if a not in chosen_set and _is_coprime_mask(a, rest_m)]
            
            if candidates:
                a = random.choice(candidates)
                new_score = current_score - _RECIP[removed] + _RECIP[a]
                delta = new_score - current_score
                
                if delta > 0 or random.random() < pow(2.718281828, delta / t):
                    # Accept move
                    chosen_set.remove(removed)
                    chosen_set.add(a)
                    current[i] = a
                    union_m = rest_m | _MASKS[a]
                    current_score = new_score
                    
                    if current_score > best_score + 1e-12:
                        best = list(current)
                        best_score = current_score
        
        # Try a random add
        candidates = [a for a in range(511, 0, -1) 
                    if a not in chosen_set and _is_coprime_mask(a, union_m)]
        if candidates:
            a = random.choice(candidates)
            new_score = current_score + _RECIP[a]
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / t):
                # Accept move
                chosen_set.add(a)
                current.append(a)
                union_m |= _MASKS[a]
                current_score = new_score
                
                if current_score > best_score + 1e-12:
                    best = list(current)
                    best_score = current_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Greedy initialization
    chosen = _greedy_from_high(deadline)
    if not chosen:
        return [511]
    
    # Phase 2: Local search
    if time.time() < deadline:
        chosen, best_score = _local_search(chosen, deadline)
    
    # Phase 3: Simulated annealing to escape local optima
    if time.time() < deadline - 5.0:
        chosen, best_score = _simulated_annealing(chosen, deadline)
    
    # Phase 4: Final local search polish
    if time.time() < deadline:
        chosen, best_score = _local_search(chosen, deadline)
    
    return chosen
# EVOLVE-BLOCK-END
