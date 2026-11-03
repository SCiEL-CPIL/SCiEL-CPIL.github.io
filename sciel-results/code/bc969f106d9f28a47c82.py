"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(U_arr[-1])
    if max_u == 0:
        return 1.0
    
    # Sumset: all pairwise sums
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # Difference set: all pairwise differences
    diffs = U_arr[:, None] - U_arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """Faster scoring using numpy, with early exit for obviously bad sets."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n < 2:
        return 0.0
    max_u = int(U_arr[-1])
    
    # For n elements, |U+U| >= 2n-1 (if sorted, min sum=0, max sum=2*max)
    # |U-U| >= 2n-1 as well
    # Quick upper bound check
    # Max possible |U-U| = 2*max_u + 1
    # Max possible |U+U| = 2*max_u + 1
    # But we need the actual ratio
    
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    diffs = U_arr[:, None] - U_arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _search_small_sets(max_n=6, max_val=30, deadline=None):
    """Exhaustive search for small sets U with size up to max_n, values up to max_val."""
    from itertools import combinations
    
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            s = _score_set_fast(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _search_structured(max_n=7, max_val=60, deadline=None):
    """Search structured families: AP with one perturbation, two perturbations, etc."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    # Family 1: {0, 1, ..., k-2, M} for various k, M
    for k in range(3, 9):
        for M in range(k, max_val + 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                s = _score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # Family 2: {0, a, b, c} with 0 < a < b < c
    # Already covered by exhaustive for n=4
    
    # Family 3: Two perturbed AP: {0, 1, ..., k-3, M, N}
    for k in range(4, 8):
        base = list(range(k - 2))  # 0, 1, ..., k-3
        for M in range(k - 1, max_val + 1):
            for N in range(M + 1, max_val + 1):
                if deadline and time.time() > deadline:
                    return best_U, best_score
                U = base + [M, N]
                if len(set(U)) == len(U):
                    s = _score_set_fast(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
    
    # Family 4: Geometric-like: {0, 1, r, r^2, ...}
    for r in range(2, 10):
        for n in range(3, 7):
            if deadline and time.time() > deadline:
                return best_U, best_score
            U = [0]
            val = 1
            for i in range(n - 1):
                U.append(val)
                val *= r
            if max(U) <= max_val * 2:
                s = _score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    return best_U, best_score

def _local_search(U, max_iters=500, deadline=None):
    """Improve set U via local search: move, add, remove elements."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max(max_u + 30, 100)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try moving one element (not 0)
        for i in range(1, len(U)):
            old_val = U[i]
            for new_val in range(1, upper_bound + 1):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_fast(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try adding a new element
        for new_val in range(1, upper_bound + 1):
            if new_val in best_U:
                continue
            new_U = best_U + [new_val]
            new_U.sort()
            s = _score_set_fast(new_U)
            if s > best_score + 1e-9:
                best_score = s
                best_U = new_U[:]
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing an element (not 0)
        for i in range(1, len(best_U)):
            new_U = best_U[:i] + best_U[i+1:]
            if len(new_U) < 2:
                continue
            s = _score_set_fast(new_U)
            if s > best_score + 1e-9:
                best_score = s
                best_U = new_U[:]
                improved = True
                break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def _simulated_annealing(U, max_iters=3000, deadline=None):
    """Simulated annealing to escape local optima."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    current_score = best_score
    current_U = U[:]
    max_u = max(U)
    upper_bound = max(max_u + 40, 120)
    
    T = 0.1
    cooling = 0.995
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Generate a neighbor
        op = random.random()
        new_U = current_U[:]
        
        if op < 0.5 and len(current_U) > 1:
            # Move an element
            i = random.randint(1, len(current_U) - 1)
            new_val = random.randint(1, upper_bound)
            new_U[i] = new_val
        elif op < 0.75 and len(current_U) < 10:
            # Add an element
            new_val = random.randint(1, upper_bound)
            if new_val not in current_U:
                new_U.append(new_val)
            else:
                continue
        elif op < 0.9 and len(current_U) > 2:
            # Remove an element
            i = random.randint(1, len(current_U) - 1)
            new_U.pop(i)
        else:
            # Swap two elements
            if len(current_U) >= 3:
                i, j = random.sample(range(1, len(current_U)), 2)
                new_U[i], new_U[j] = new_U[j], new_U[i]
            else:
                continue
        
        new_U = sorted(set(new_U))
        if len(new_U) < 2 or 0 not in new_U:
            continue
        
        s = _score_set_fast(new_U)
        
        # Accept if better or with probability exp((s - current_score)/T)
        if s > current_score:
            current_U = new_U
            current_score = s
            if s > best_score:
                best_score = s
                best_U = new_U[:]
        else:
            delta = current_score - s
            if delta > 0 and random.random() < math.exp(-delta / T):
                current_U = new_U
                current_score = s
        
        T *= cooling
        
    return sorted(set(best_U)), best_score

def _construct_heuristic_sets(deadline=None):
    """Construct sets based on known good structures."""
    candidates = []
    
    # Known good small sets to test
    test_sets = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 5],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 4, 6],
        [0, 1, 4, 10],
        [0, 2, 3, 7],
        [0, 1, 3, 8],
        [0, 1, 5, 8],
        [0, 2, 5, 7],
        [0, 1, 4, 9],
        [0, 1, 6, 7],
        [0, 2, 3, 8],
        [0, 1, 3, 9],
        [0, 1, 5, 9],
        [0, 2, 5, 9],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 18],
        [0, 2, 3, 8, 11],
        [0, 1, 5, 8, 12],
    ]
    candidates.extend(test_sets)
    
    # Structure: {0, 1, ..., k-2, M}
    for k in range(3, 10):
        for M in range(k, 80):
            if deadline and time.time() > deadline:
                break
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)
    
    # Structure: {0, a, b, a+b}
    for a in range(1, 20):
        for b in range(a + 1, 40):
            if deadline and time.time() > deadline:
                break
            U = [0, a, b, a + b]
            if len(set(U)) == 4:
                candidates.append(U)
    
    # Structure: {0, a, 2a, 3a, M}
    for a in range(1, 15):
        for M in range(4 * a, 4 * a + 40):
            if deadline and time.time() > deadline:
                break
            U = [0, a, 2*a, 3*a, M]
            if len(set(U)) == 5:
                candidates.append(U)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (150s limit, 20s buffer)
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # 1. Exhaustive search for small sets (n up to 5, values up to 35)
    try:
        U_sol, s_sol = _search_small_sets(max_n=5, max_val=35, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
    
    # 2. Structured search
    try:
        U_sol, s_sol = _search_structured(max_n=7, max_val=60, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
    
    # 3. Heuristic candidates
    try:
        candidates = _construct_heuristic_sets(deadline=deadline)
        for U in candidates:
            if deadline and time.time() > deadline:
                break
            s = _score_set_fast(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    except Exception:
        pass
    
    # 4. Local search on best found
    if time.time() < deadline:
        try:
            best_U, best_score = _local_search(best_U, max_iters=500, deadline=deadline)
        except Exception:
            pass
    
    # 5. Simulated annealing for further improvement
    if time.time() < deadline:
        try:
            best_U2, best_score2 = _simulated_annealing(best_U, max_iters=3000, deadline=deadline)
            if best_score2 > best_score:
                best_score = best_score2
                best_U = best_U2
        except Exception:
            pass
    
    # 6. Final local search polish
    if time.time() < deadline:
        try:
            best_U, best_score = _local_search(best_U, max_iters=200, deadline=deadline)
        except Exception:
            pass
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
