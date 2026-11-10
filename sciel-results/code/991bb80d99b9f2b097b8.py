"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from functools import lru_cache
import random

def _compute_scores(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A = list(A)
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _generate_params(b, n=20, max_val=10000):
    """Generate a list of (e, k) pairs that produce distinct values <= max_val."""
    powers = []
    val = 1
    while val <= max_val:
        powers.append(val)
        val *= b
    
    small_multipliers = [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    candidates = []
    for e, p_val in enumerate(powers):
        for k in small_multipliers:
            v = p_val * k
            if v <= max_val:
                candidates.append((e, k, v))
            else:
                break
                
    candidates.sort(key=lambda x: x[2])
    
    params = []
    used_values = set()
    count = 0
    for e, k, v in candidates:
        if v not in used_values:
            used_values.add(v)
            params.append((e, k))
            count += 1
            if count == n:
                break
            
    if count < n:
        return None
        
    return params

def _params_to_set(params, b):
    """Convert (e, k) params to set of integers."""
    A = set()
    for e, k in params:
        v = (b ** e) * k
        A.add(v)
    return sorted(A)

def _local_search_params(params, b, max_iter=50000, max_val=10000):
    """Local search in the parameter space (e, k)."""
    current_params = params[:]
    current_set = _params_to_set(current_params, b)
    current_score = _compute_scores(current_set)
    best_params = current_params[:]
    best_score = current_score
    best_set = current_set[:]
    
    small_multipliers = [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    for _ in range(max_iter):
        i = random.randint(0, len(current_params) - 1)
        e_old, k_old = current_params[i]
        
        proposed_params = current_params[:]
        
        if random.random() < 0.5:
            delta_e = random.choice([-2, -1, 1, 2])
            e_new = e_old + delta_e
            if e_new < 0:
                continue
            proposed_params[i] = (e_new, k_old)
        else:
            k_new = random.choice(small_multipliers)
            if k_new == k_old:
                continue
            proposed_params[i] = (e_old, k_new)
            
        new_set = _params_to_set(proposed_params, b)
        if len(new_set) != len(current_params):
            continue
            
        new_score = _compute_scores(new_set)
        
        if new_score < best_score:
            best_score = new_score
            best_set = new_set
            best_params = proposed_params[:]
            current_params = proposed_params[:]
            current_set = new_set
            current_score = new_score
            
    return best_set, best_score

def construct():
    best_set = None
    best_score = float('inf')
    
    for b in [2, 3, 5]:
        params = _generate_params(b, n=20, max_val=10000)
        if params is None:
            continue
            
        set_candidate, score_candidate = _local_search_params(params, b, max_iter=50000, max_val=10000)
        
        if score_candidate < best_score:
            best_score = score_candidate
            best_set = set_candidate
            
    if best_set is None:
        best_set = list(range(1, 21))
        
    return best_set

# EVOLVE-BLOCK-END
