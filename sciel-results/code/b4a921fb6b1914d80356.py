import math
import time
import random

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
        
    # Score = 1 + log(ratio) / log(2 * mx + 1)
    # Using natural log, base doesn't matter for ratio
    try:
        val = 1.0 + math.log(ratio) / math.log(2 * mx + 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return val

def _random_set(n, max_val):
    # Must contain 0
    U = {0}
    while len(U) < n:
        U.add(random.randint(1, max_val))
    return list(U)

def _hill_climb(U, max_steps=5000, deadline=None):
    """Proper hill climbing with reversion."""
    best_U = list(U)
    best_score = _score(best_U)
    current_U = list(U)
    current_score = best_score
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random modification
        if random.random() < 0.4:
            # Add element
            current_U.append(random.randint(1, 200))
        elif random.random() < 0.5 and len(current_U) > 2:
            # Remove element (keep 0)
            idx = random.randint(1, len(current_U)-1)
            current_U.pop(idx)
        elif len(current_U) > 1:
            # Swap an element
            idx = random.randint(1, len(current_U)-1)
            current_U[idx] = random.randint(1, 200)
        else:
            # Add element
            current_U.append(random.randint(1, 200))
        
        new_score = _score(current_U)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = list(current_U)
        else:
            # Revert to previous state
            if random.random() < 0.4:
                # Add element was last action
                if len(current_U) > len(best_U):
                    current_U.pop()
                elif len(current_U) < len(best_U) and len(current_U) > 2:
                    # Can't easily revert removal, restart from best
                    current_U = list(best_U)
                else:
                    current_U = list(best_U)
            else:
                current_U = list(best_U)
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_U = [0, 1, 3]
    
    # Check some known good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 6],
        [0, 1, 4, 10],
        [0, 2, 3, 7],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 8],
        [0, 1, 5, 6],
        [0, 1, 3, 12],
        [0, 1, 4, 16],
        [0, 1, 5, 20],
        [0, 2, 4, 8],
        [0, 1, 3, 6, 12],
        [0, 1, 4, 8, 16],
        [0, 1, 3, 9, 27, 81],
        [0, 1, 2, 5, 13],
        [0, 1, 3, 7, 15],
        [0, 2, 5, 10],
        [0, 1, 4, 7, 14],
    ]
    
    for U in candidates:
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Random search with hill climbing
    for n in [3, 4, 5, 6]:
        for _ in range(3000):
            if time.time() > deadline:
                break
            U = _random_set(n, 100)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
        
        # Hill climb from current best
        if time.time() < deadline - 5:
            U_hill, s_hill = _hill_climb(best_U, max_steps=2000, deadline=deadline - 5)
            if s_hill > best_score:
                best_score = s_hill
                best_U = U_hill
    
    # Final hill climb from best
    if time.time() < deadline - 2:
        U_hill, s_hill = _hill_climb(best_U, max_steps=5000, deadline=deadline - 1)
        if s_hill > best_score:
            best_score = s_hill
            best_U = U_hill
    
    return best_U
