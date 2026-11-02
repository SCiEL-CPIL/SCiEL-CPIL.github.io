"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set using numpy for speed
    U_list = sorted(U_set)
    U_arr = list(U_list)
    
    # Use set operations for efficiency
    sumset = set()
    diffset = set()
    
    for a in U_arr:
        for b in U_arr:
            sumset.add(a + b)
            diffset.add(a - b)
            
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _search_small_sets(max_n=6, max_val=35, deadline=None):
    """Exhaustive search for small sets U."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        # Generate combinations of n-1 numbers from 1 to max_val
        # For n=6, max_val=35: C(35,5) = 324,632 - manageable
        # For n=7, we'd need a tighter max_val
        
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            U.sort()
            
            # Calculate score
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _construct_heuristic_sets(deadline=None):
    """Construct sets based on known good structures."""
    candidates = []
    
    # Structure 1: Almost AP
    # {0, 1, ..., k-2, M}
    for k in range(3, 10):
        for M in range(k, 50):
            if deadline and time.time() > deadline:
                break
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)
                
    # Structure 2: Golomb rulers (might have R < 1, but good to check)
    # Known optimal Golomb rulers
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
    ]
    candidates.extend(known_gr)
    
    # Structure 3: Randomized local search on AP
    random.seed(42)
    for _ in range(50):
        if deadline and time.time() > deadline:
            break
        n = random.randint(3, 8)
        base_ap = list(range(n))
        # Perturb one element
        i = random.randint(1, n - 1)
        new_val = random.randint(1, 60)
        U = base_ap[:]
        U[i] = new_val
        if len(set(U)) == len(U) and 0 in U:
            candidates.append(sorted(set(U)))
            
    return candidates

def _local_search(U, max_iters=300, deadline=None):
    """Improve set U via local search with add/remove/move operations."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 30
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try moving one element
        for i in range(len(U)):
            if improved:
                break
            old_val = U[i]
            if old_val == 0:
                continue
            for new_val in range(1, upper_bound + 1):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try adding a new element
        for new_val in range(1, upper_bound + 1):
            if new_val in best_U:
                continue
            new_U = best_U[:] + [new_val]
            new_U.sort()
            s = _score_set(new_U)
            if s > best_score + 1e-10:
                best_score = s
                best_U = new_U[:]
                improved = True
                break
        
        if improved:
            continue
            
        # Try removing an element (if size > 3)
        if len(best_U) > 3:
            for i in range(1, len(best_U)):
                new_U = best_U[:i] + best_U[i+1:]
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # 1. Exhaustive search for small sets
    # n=5, max_val=40: C(40,4) = 91,390 - fast
    # n=6, max_val=35: C(35,5) = 324,632 - feasible
    try:
        U_sol, s_sol = _search_small_sets(max_n=6, max_val=35, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
        
    # 2. Heuristic candidates
    candidates = _construct_heuristic_sets(deadline=deadline)
    for U in candidates:
        if deadline and time.time() > deadline:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # 3. Local search on best found - more iterations
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=300, deadline=deadline)
        
    # 4. Final polish pass with tighter tolerance
    if time.time() < deadline - 5:
        # Try a few more local search passes from different starting points
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb and re-search
            perturbed = best_U[:]
            if len(perturbed) > 3:
                idx = random.randint(1, len(perturbed) - 1)
                old_val = perturbed[idx]
                new_val = old_val + random.randint(-5, 5)
                if new_val > 0 and new_val not in perturbed:
                    perturbed[idx] = new_val
                    perturbed.sort()
                    perturbed = list(set(perturbed))
                    if 0 in perturbed:
                        s = _score_set(perturbed)
                        if s > best_score:
                            best_score = s
                            best_U = perturbed[:]
            
            if time.time() < deadline - 5:
                best_U, best_score = _local_search(best_U, max_iters=100, deadline=deadline)
        
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
