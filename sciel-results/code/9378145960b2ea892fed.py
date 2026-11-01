import time
import math
import random

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute sumset and difference set
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    global _START_TIME
    deadline = _START_TIME + _TIME_LIMIT
    
    # Start with a good initial set
    # Try arithmetic progression first as baseline
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try various structured sets
    candidate_sets = []
    
    # Arithmetic progressions
    for n in range(3, 50):
        U = list(range(n))
        candidate_sets.append(U)
    
    # Geometric-like
    for base in [2, 3, 4, 5]:
        U = [0]
        x = 1
        while x < 10000:
            U.append(x)
            x *= base
        candidate_sets.append(U)
    
    # Mixed structures
    for n in range(3, 30):
        for gap in [1, 2, 3, 5, 7, 10]:
            U = [0]
            x = gap
            for i in range(1, n):
                U.append(x)
                x += gap
            candidate_sets.append(U)
    
    # Evaluate all candidates
    for U in candidate_sets:
        if time.time() > deadline - 5:
            break
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Local search: try to improve best_U
    # Strategy: try adding/removing elements, or shifting elements
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        U = sorted(set(best_U))
        n = len(U)
        max_u = max(U) if U else 0
        
        # Try adding a new element
        for new_elem in range(1, max_u + 20):
            if new_elem not in U:
                U_new = sorted(U + [new_elem])
                s = _compute_score(U_new)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = U_new
                    improved = True
                    break
            if time.time() > deadline - 10:
                break
        
        # Try removing an element (except 0)
        if time.time() > deadline - 10:
            break
        for i in range(1, len(U)):
            U_new = U[:i] + U[i+1:]
            if len(U_new) < 2:
                continue
            s = _compute_score(U_new)
            if s > best_score + 1e-12:
                best_score = s
                best_U = U_new
                improved = True
                break
        
        # Try replacing an element
        if time.time() > deadline - 10:
            break
        for i in range(1, len(U)):
            for new_val in range(1, max_u + 10):
                if new_val == U[i] or new_val in U:
                    continue
                U_new = sorted(set(U[:i] + U[i+1:] + [new_val]))
                if len(U_new) < 2:
                    continue
                s = _compute_score(U_new)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = U_new
                    improved = True
                    break
            if improved or time.time() > deadline - 10:
                break
    
    # Try some more sophisticated constructions
    if time.time() < deadline - 5:
        # Try sets with elements at powers and multiples
        for p in [2, 3, 5]:
            for m in [2, 3, 4, 5]:
                U = [0]
                for i in range(1, 8):
                    U.append(p ** i)
                    if len(U) > 15:
                        break
                candidate = sorted(set(U))
                s = _compute_score(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate
    
    return list(best_U)
