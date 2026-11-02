"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _score_set(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    
    U_set = set(U)
    n = len(U_set)
    
    # Compute |U+U| and |U-U|
    sum_set = set()
    diff_set = set()
    
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    
    ratio = len(diff_set) / len(sum_set)
    score = 1 + math.log(ratio) / log_denom
    return score

def _generate_primes(limit):
    """Generate primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def _greedy_construct(primes, max_steps=100):
    """
    Greedily construct U by adding the prime that maximizes the score.
    Uses the smallest primes first to keep max(U) small.
    """
    U = [0]
    used_primes = set()
    best_score = _score_set(U)
    
    for _ in range(max_steps):
        best_prime = None
        best_new_score = best_score
        
        # Iterate through primes in ascending order
        for p in primes:
            if p in used_primes:
                continue
            
            new_U = U + [U[-1] + p]
            new_score = _score_set(new_U)
            
            # We require a strict improvement to avoid oscillation or stagnation
            if new_score > best_new_score + 1e-10:
                best_new_score = new_score
                best_prime = p
        
        if best_prime is None:
            break
        
        U.append(U[-1] + best_prime)
        used_primes.add(best_prime)
        best_score = best_new_score
        
        # Early stopping if score starts to drop significantly or stabilizes
        # The score is bounded and decreases as max(U) grows if ratio doesn't improve enough
        if best_score < 1.0:
            # Revert if we went below 1? No, we only accept improvements.
            # If we can't improve, we break.
            pass
    
    return U

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound.
    
    Uses a fast greedy prime selection strategy.
    """
    # Generate a sufficient pool of primes
    primes = _generate_primes(1000)
    
    # Run the greedy constructor
    best_U = _greedy_construct(primes, max_steps=100)
    
    return best_U
# EVOLVE-BLOCK-END
