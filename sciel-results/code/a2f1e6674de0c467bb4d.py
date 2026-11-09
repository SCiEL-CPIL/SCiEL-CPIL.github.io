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

def _exhaustive_search(max_val, max_n, deadline):
    best_score = 0.0
    best_U = [0, 1, 3] # Default
    
    # We always include 0. We choose n-1 elements from 1..max_val
    for n in range(3, max_n + 1):
        # Choose n-1 elements from 1 to max_val
        elements = range(1, max_val + 1)
        for combo in combinations(elements, n - 1):
            if time.time() > deadline:
                return best_score, best_U
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
                
    return best_score, best_U

def _hill_climb(U, max_steps, deadline):
    best_U = list(U)
    best_score = _score(best_U)
    current_U = list(U)
    current_score = best_score
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Random modification
        if random.random() < 0.3:
            # Add element
            current_U.append(random.randint(1, 100))
        elif random.random() < 0.5 and len(current_U) > 2:
            # Remove element (keep 0)
            idx = random.randint(1, len(current_U)-1)
            current_U.pop(idx)
        elif len(current_U) > 1:
            # Swap an element
            idx = random.randint(1, len(current_U)-1)
            current_U[idx] = random.randint(1, 100)
        else:
            current_U.append(random.randint(1, 100))
            
        new_score = _score(current_U)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = list(current_U)
        else:
            # Revert
            if random.random() < 0.5:
                current_U = list(best_U)
            else:
                # Try to revert the last move roughly
                if len(current_U) > len(best_U):
                    current_U.pop()
                elif len(current_U) < len(best_U):
                    current_U = list(best_U)
                else:
                    current_U = list(best_U)
                    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = 0.0
    best_U = [0, 1, 3]
    
    # Phase 1: Exhaustive search for small sets
    # We can afford to search up to n=7 or 8 with small max_val
    # Let's try max_val=30, max_n=7
    # Combinations of 6 from 30 is ~593775. Might be too slow in Python if _score is slow.
    # Let's start with smaller bounds and increase if time permits.
    
    s, U = _exhaustive_search(20, 6, deadline)
    if s > best_score:
        best_score = s
        best_U = U
        
    s, U = _exhaustive_search(30, 5, deadline)
    if s > best_score:
        best_score = s
        best_U = U
        
    s, U = _exhaustive_search(40, 4, deadline)
    if s > best_score:
        best_score = s
        best_U = U

    # Phase 2: Hill Climbing from best found
    if time.time() < deadline - 5:
        U_hill, s_hill = _hill_climb(best_U, max_steps=5000, deadline=deadline - 2)
        if s_hill > best_score:
            best_score = s_hill
            best_U = U_hill

    return best_U
