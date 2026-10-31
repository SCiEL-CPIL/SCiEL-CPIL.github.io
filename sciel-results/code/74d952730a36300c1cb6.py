"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _score(U):
    """Compute the lower bound score for a finite set U containing 0."""
    if not U or 0 not in U:
        return -1.0
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return -1.0
    max_u = max(U)
    if max_u == 0:
        return -1.0
    
    # Compute |U+U| and |U-U|
    sum_set = set()
    diff_set = set()
    for i in range(n):
        for j in range(n):
            sum_set.add(U[i] + U[j])
            diff_set.add(U[i] - U[j])
    
    sum_size = len(sum_set)
    diff_size = len(diff_set)
    
    if sum_size == 0:
        return -1.0
    
    log_arg = diff_size / sum_size
    if log_arg <= 0:
        return -1.0
    
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1.0
    
    return 1 + math.log(log_arg) / denom

def _generate_candidate(n, max_val):
    """Generate a random set of n nonnegative integers containing 0, all <= max_val."""
    if n <= 1:
        return [0]
    # Pick n-1 random distinct values from 1..max_val
    if max_val < n - 1:
        # Not enough values, just use 0..n-1
        return list(range(n))
    vals = [0] + random.sample(range(1, max_val + 1), n - 1)
    return sorted(vals)

def _local_search(U, max_val, time_limit):
    """Local search: try adding, removing, or replacing elements."""
    best_U = list(U)
    best_score = _score(best_U)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        n = len(best_U)
        
        # Try adding a random element
        if random.random() < 0.3 and n < 50:
            new_val = random.randint(1, max_val)
            if new_val not in best_U:
                candidate = sorted(best_U + [new_val])
                s = _score(candidate)
                if s > best_score:
                    best_U = candidate
                    best_score = s
                    improved = True
        
        # Try removing a random element (keep 0)
        if random.random() < 0.2 and n > 2:
            removable = [i for i in range(n) if best_U[i] != 0]
            if removable:
                idx = random.choice(removable)
                candidate = best_U[:idx] + best_U[idx+1:]
                s = _score(candidate)
                if s > best_score:
                    best_U = candidate
                    best_score = s
                    improved = True
        
        # Try replacing a random element (not 0)
        if random.random() < 0.5 and n > 1:
            replaceable = [i for i in range(n) if best_U[i] != 0]
            if replaceable:
                idx = random.choice(replaceable)
                new_val = random.randint(1, max_val)
                if new_val not in best_U:
                    candidate = best_U[:idx] + [new_val] + best_U[idx+1:]
                    candidate = sorted(candidate)
                    s = _score(candidate)
                    if s > best_score:
                        best_U = candidate
                        best_score = s
                        improved = True
        
        # Try perturbing a random element
        if random.random() < 0.3 and n > 1:
            perturbable = [i for i in range(n) if best_U[i] != 0]
            if perturbable:
                idx = random.choice(perturbable)
                delta = random.randint(-max_val//10, max_val//10)
                new_val = max(1, min(max_val, best_U[idx] + delta))
                if new_val not in best_U:
                    candidate = best_U[:idx] + [new_val] + best_U[idx+1:]
                    candidate = sorted(candidate)
                    s = _score(candidate)
                    if s > best_score:
                        best_U = candidate
                        best_score = s
                        improved = True
        
        if not improved:
            break
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Try various sizes and max values
    sizes_to_try = [3, 5, 8, 12, 16, 20, 25, 30]
    max_vals = [10, 20, 50, 100, 200, 500]
    
    random.seed(42)
    
    for max_val in max_vals:
        if time.time() > deadline - 10:
            break
        for n in sizes_to_try:
            if time.time() > deadline - 10:
                break
            
            # Multiple restarts
            for restart in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Generate initial candidate
                if n <= max_val:
                    U = _generate_candidate(n, max_val)
                    s = _score(U)
                    if s > best_score:
                        best_U = U
                        best_score = s
                    
                    # Local search with remaining time
                    remaining = deadline - 10 - time.time()
                    if remaining > 2:
                        U_ls, s_ls = _local_search(U, max_val, min(remaining, 8))
                        if s_ls > best_score:
                            best_U = U_ls
                            best_score = s_ls
                else:
                    break
    
    # Final check
    if time.time() > deadline - 5:
        return best_U
    
    return best_U
# EVOLVE-BLOCK-END
