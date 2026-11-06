"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

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

def _greedy_init():
    """Greedy initialization: pick largest coprime numbers first."""
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    return chosen

def _local_search(chosen):
    """Swap-based local search to improve the score."""
    best = list(chosen)
    best_score = _score(best)
    
    while not _time_exhausted():
        improved = False
        # Try all possible swaps: remove one element, try adding another
        for i in range(len(best)):
            if _time_exhausted():
                break
            removed = best[i]
            candidate_set = best[:i] + best[i+1:]
            
            # Try adding each possible candidate
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
        
        # Also try adding elements without removing
        if not improved:
            for a in range(511, 0, -1):
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
        
        # Try removing elements (if it helps, though unlikely)
        if not improved:
            for i in range(len(best)):
                if _time_exhausted():
                    break
                candidate_set = best[:i] + best[i+1:]
                new_score = _score(candidate_set)
                
                # Only remove if it somehow helps (shouldn't happen, but check)
                if new_score > best_score + 1e-15:
                    best = candidate_set
                    best_score = new_score
                    improved = True
        
        if not improved:
            break
    
    return best

def construct():
    # Start with greedy initialization
    chosen = _greedy_init()
    
    # Apply local search to improve
    chosen = _local_search(chosen)
    
    return chosen
# EVOLVE-BLOCK-END
