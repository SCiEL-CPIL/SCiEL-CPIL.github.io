"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the lower bound score for a set U."""
    U = sorted(set(U))
    if len(U) < 2 or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute |U + U| and |U - U|
    U_arr = np.array(U)
    sum_set = np.unique(U_arr[:, None] + U_arr[None, :])
    diff_set = np.unique(U_arr[:, None] - U_arr[None, :])
    
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    denom = 2 * max_u + 1
    if denom <= 1:
        return 0.0
    
    score = 1.0 + math.log(n_diff / n_sum) / math.log(denom)
    return score

def _random_set(n, max_val):
    """Generate a random set of size n containing 0, with values in [0, max_val]."""
    if n <= 1:
        return [0]
    # Pick n-1 random values from [1, max_val], ensure 0 is included
    others = set(np.random.choice(range(1, max_val + 1), size=n - 1, replace=False))
    return [0] + list(others)

def _greedy_improve(U, max_val, time_limit):
    """Try to improve U by adding or removing elements."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U.sort()
    
    best_score = _score_set(U)
    best_U = list(U)
    start = time.time()
    
    # Try adding elements
    for _ in range(50):
        if time.time() - start > time_limit:
            break
        # Try adding a random element
        new_val = np.random.choice(range(1, max_val + 1), size=1, replace=False)[0]
        if new_val not in U:
            new_U = U + [new_val]
            new_score = _score_set(new_U)
            if new_score > best_score:
                best_score = new_score
                best_U = new_U
    
    # Try removing elements (keep 0)
    for _ in range(50):
        if time.time() - start > time_limit:
            break
        if len(U) <= 2:
            break
        # Remove a random non-zero element
        idx = np.random.choice(range(1, len(U)), size=1, replace=False)[0]
        new_U = U[:idx] + U[idx+1:]
        new_score = _score_set(new_U)
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with a good initial set
    # The paper mentions constructions of size 2003 and 54265
    # Let's try a few different sizes
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try various sizes and max values
    sizes_to_try = [5, 10, 20, 50, 100, 200, 500]
    max_vals_to_try = [10, 50, 100, 500, 1000, 5000]
    
    for n in sizes_to_try:
        if time.time() > deadline:
            break
        for max_val in max_vals_to_try:
            if time.time() > deadline:
                break
            
            # Try a few random initializations
            for attempt in range(3):
                if time.time() > deadline:
                    break
                
                U = _random_set(n, max_val)
                score = _score_set(U)
                
                if score > best_score:
                    best_score = score
                    best_U = U
                
                # Try greedy improvement
                time_remaining = deadline - time.time()
                if time_remaining > 2:
                    improved_U, improved_score = _greedy_improve(
                        U, max_val, min(5, time_remaining)
                    )
                    if improved_score > best_score:
                        best_score = improved_score
                        best_U = improved_U
    
    # Also try structured sets
    # Arithmetic progression
    for n in [5, 10, 20, 50]:
        if time.time() > deadline:
            break
        for d in [1, 2, 3, 5]:
            if time.time() > deadline:
                break
            U = list(range(0, n * d, d))
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Geometric progression
    for base in [2, 3, 4]:
        if time.time() > deadline:
            break
        U = [0]
        val = 1
        while val <= 10000:
            U.append(val)
            val *= base
        score = _score_set(U)
        if score > best_score:
            best_score = score
            best_U = U
    
    return best_U
# EVOLVE-BLOCK-END
