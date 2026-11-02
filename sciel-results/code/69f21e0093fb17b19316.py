"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

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

def _optimize_gaps(initial_gaps, max_iterations=5000, neighbor_range=10):
    """
    Local search optimization on the list of gaps.
    """
    current_gaps = initial_gaps[:]
    current_U = [0]
    for g in current_gaps:
        current_U.append(current_U[-1] + g)
    current_score = _score_set(current_U)
    
    primes = _generate_primes(1000)
    
    for _ in range(max_iterations):
        improved = False
        # Try modifying each gap
        for i in range(len(current_gaps)):
            old_g = current_gaps[i]
            # Candidates: nearby integers, and nearby primes
            candidates = set()
            for delta in range(-neighbor_range, neighbor_range + 1):
                new_g = old_g + delta
                if new_g > 0:
                    candidates.add(new_g)
            
            # Add some random primes
            for p in primes:
                if abs(p - old_g) < neighbor_range * 2:
                    candidates.add(p)
            
            best_candidate = old_g
            best_score = current_score
            
            for new_g in candidates:
                if new_g == old_g:
                    continue
                new_gaps = current_gaps[:]
                new_gaps[i] = new_g
                # Reconstruct U
                new_U = [0]
                for g in new_gaps:
                    new_U.append(new_U[-1] + g)
                
                # Check for duplicates in U (should not happen if gaps > 0)
                if len(set(new_U)) != len(new_U):
                    continue
                
                new_score = _score_set(new_U)
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = new_g
            
            if best_candidate != old_g:
                current_gaps[i] = best_candidate
                current_score = best_score
                improved = True
                # Reconstruct U for next iteration
                current_U = [0]
                for g in current_gaps:
                    current_U.append(current_U[-1] + g)
        
        # Try adding a new gap at the end
        if len(current_gaps) < 100: # Limit size to avoid timeout
            # Try small primes
            for p in primes[:20]:
                new_gaps = current_gaps + [p]
                new_U = current_U + [current_U[-1] + p]
                new_score = _score_set(new_U)
                if new_score > current_score:
                    current_gaps = new_gaps
                    current_U = new_U
                    current_score = new_score
                    improved = True
        
        # Try removing the last gap
        if len(current_gaps) > 1:
            new_gaps = current_gaps[:-1]
            new_U = current_U[:-1]
            new_score = _score_set(new_U)
            if new_score > current_score:
                current_gaps = new_gaps
                current_U = new_U
                current_score = new_score
                improved = True
        
        if not improved:
            break
            
    return current_U, current_score

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound.
    
    Uses a local search heuristic on prime-step gaps.
    """
    # Initial set of gaps: first few primes
    initial_gaps = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    best_U, best_score = _optimize_gaps(initial_gaps)
    
    return best_U
# EVOLVE-BLOCK-END
