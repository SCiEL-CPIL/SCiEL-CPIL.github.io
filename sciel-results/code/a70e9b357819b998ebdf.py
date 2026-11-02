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
    
    # Compute sumset and difference set
    # For small n, direct computation is fast
    # For larger n, we might need numpy, but optimal sets are usually small
    sumset = set()
    diffset = set()
    
    U_list = list(U_set)
    for a in U_list:
        for b in U_list:
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

def _search_small_sets(max_n=8, max_val=25, deadline=None):
    """Exhaustive search for small sets U."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        # Generate combinations of n-1 numbers from 1 to max_val
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

def _local_search(U, max_iters=1000, deadline=None, upper_bound=100):
    """Improve set U via local search with add/remove/move operations."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    
    # Dynamic upper bound based on current max
    current_max = max(best_U)
    ub = max(upper_bound, current_max + 20)
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # 1. Try moving one element to a new value
        # Limit search space for moves to keep it fast
        for i in range(len(best_U)):
            if improved:
                break
            old_val = best_U[i]
            if old_val == 0:
                continue
            # Search a window around old_val and some random spots
            # To be efficient, we don't scan 1..ub every time
            # We scan a local neighborhood and a few global spots
            candidates = set()
            # Local neighborhood
            for delta in range(-5, 6):
                v = old_val + delta
                if 1 <= v <= ub and v != old_val:
                    candidates.add(v)
            # Global random spots (to escape local minima)
            for _ in range(5):
                v = random.randint(1, ub)
                if v != old_val:
                    candidates.add(v)
            
            for new_val in candidates:
                if new_val in best_U:
                    continue
                new_U = best_U[:]
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
            
        # 2. Try adding a new element
        # Scan a subset of possible values
        add_candidates = set()
        # Neighbors of existing elements
        for v in best_U:
            for delta in range(-2, 3):
                nv = v + delta
                if 1 <= nv <= ub and nv not in best_U:
                    add_candidates.add(nv)
        # Random spots
        for _ in range(10):
            nv = random.randint(1, ub)
            if nv not in best_U:
                add_candidates.add(nv)
        
        for new_val in add_candidates:
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
            
        # 3. Try removing an element (if size > 3)
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
            # If no improvement, try a larger jump or break
            # To avoid getting stuck, we can randomize the search slightly
            if random.random() < 0.1:
                 # Try a random move even if it doesn't improve immediately? 
                 # No, local search usually requires improvement. 
                 # Let's just break if no improvement found in this iteration
                 pass
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # 1. Exhaustive search for small sets
    # This is the most reliable way to find high-score small sets
    try:
        # n=8, max_val=25: C(25, 7) = 480,700. This is fast in Python if optimized or in C.
        # Pure Python might take ~1-2 seconds for this.
        U_sol, s_sol = _search_small_sets(max_n=8, max_val=25, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
        
    # 2. Local search on the best found
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=500, deadline=deadline)
        
    # 3. Multi-start local search
    # Try local search from a few other good starting points
    starting_points = [
        [0, 1, 3, 7],
        [0, 1, 4, 10],
        [0, 2, 5, 11],
        [0, 1, 4, 9, 15],
        [0, 2, 5, 10, 16],
    ]
    
    for sp in starting_points:
        if time.time() > deadline - 10:
            break
        # Ensure sp is valid
        if 0 in sp and len(sp) == len(set(sp)):
            U_candidate, s_candidate = _local_search(sp, max_iters=200, deadline=deadline)
            if s_candidate > best_score:
                best_score = s_candidate
                best_U = U_candidate

    # 4. Final polish
    if time.time() < deadline - 5:
        # One more pass with tighter tolerance or different parameters
        best_U, best_score = _local_search(best_U, max_iters=200, deadline=deadline)
        
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
