"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _score_set(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0.0

    # Compute |U+U| and |U-U|
    sum_set = set()
    diff_set = set()
    
    U_list = list(U_set)
    for a in U_list:
        for b in U_list:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0 # log(1)=0, so score is 1. If ratio < 1, score < 1.
                   # But we start with score 1? 
                   # For n=1, U={0}, sum={0}, diff={0}, ratio=1, score=1.
                   # For n=2, U={0,p}, sum={0,p,2p}, diff={-p,0,p}. Ratio 1. Score 1.
                   # So we need ratio > 1 to get score > 1.
    
    score = 1 + math.log(ratio) / log_denom
    return score

def _generate_primes(limit):
    """Generate primes up to limit."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def _greedy_construct(max_steps=150, prime_limit=10000):
    """
    Greedy construction of U by adding prime gaps one by one.
    """
    primes = _generate_primes(prime_limit)
    if not primes:
        return [0]
        
    U = [0]
    current_score = _score_set(U)
    
    prime_idx = 0
    window_size = 100
    
    for step in range(max_steps):
        if prime_idx + window_size > len(primes):
            break
            
        best_score = current_score
        best_U = None
        best_idx_in_window = -1
        
        # Try each prime in the window
        for i in range(window_size):
            p = primes[prime_idx + i]
            new_U = U + [U[-1] + p]
            new_score = _score_set(new_U)
            
            if new_score > best_score:
                best_score = new_score
                best_U = new_U
                best_idx_in_window = i
        
        if best_U is None:
            break
        else:
            U = best_U
            current_score = best_score
            # Advance index past the chosen prime
            prime_idx += best_idx_in_window + 1
            
    return U

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound.
    """
    return _greedy_construct()
# EVOLVE-BLOCK-END
