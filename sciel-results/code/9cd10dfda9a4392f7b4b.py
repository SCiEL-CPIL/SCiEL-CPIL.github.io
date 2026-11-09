import math
import time
import random
from itertools import combinations

def _score(U):
    U = sorted(list(set(U)))
    if 0 not in U:
        return 0.0
    if len(U) < 2:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    # Calculate sumset and difference set
    plus_set = set()
    minus_set = set()
    for a in U:
        for b in U:
            plus_set.add(a + b)
            minus_set.add(a - b)
            
    n_plus = len(plus_set)
    n_minus = len(minus_set)
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
        
    try:
        val = 1.0 + math.log(ratio) / math.log(2 * mx + 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return val

def _exhaustive_search(max_val=25, max_n=5):
    """Exhaustively search for optimal small sets."""
    best_score = 0.0
    best_U = [0, 1, 2]
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Exhaustive search over small sets
    best_U, best_score = _exhaustive_search(max_val=30, max_n=5)
    
    # Also try n=6 with smaller range
    if time.time() < deadline - 10:
        for combo in combinations(range(1, 16), 5):
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
    
    return best_U
