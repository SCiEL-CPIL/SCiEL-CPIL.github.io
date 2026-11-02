"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
import random
import itertools

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score 1 + log(|U-U|/|U+U|)/log(2 max U + 1) for a sorted list U."""
    if not U:
        return 0.0
    max_u = U[-1]
    n = len(U)
    if n < 2:
        return 1.0
    if max_u == 0:
        return 1.0
    
    # Use FFT for efficiency
    size = 2 * max_u + 1
    v = np.zeros(size, dtype=np.float64)
    v[U] = 1.0
    
    # Sumset: v * v
    conv_sum = np.convolve(v, v, mode='full')
    # The sumset U+U has elements in range [0, 2*max_u]
    # conv_sum[i] is the number of pairs (a,b) with a+b=i
    # |U+U| is the number of i where conv_sum[i] > 0
    sumset_card = np.sum(conv_sum > 0.5)
    
    # Difference set: U - U has elements in range [-max_u, max_u]
    w = v[::-1]
    conv_diff = np.convolve(w, v, mode='full')
    # conv_diff has length 2*size - 1 = 4*max_u + 1
    # We need indices from max_u to 3*max_u (corresponding to d from -max_u to max_u)
    diff_part = conv_diff[max_u : 3*max_u + 1]
    diffset_card = np.sum(diff_part > 0.5)
    
    if sumset_card == 0 or diffset_card == 0:
        return 1.0
    
    ratio = diffset_card / sumset_card
    if ratio <= 0:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    score = 1.0 + log_ratio / log_denom
    return score

def _local_search(U, deadline, max_iter=5000):
    """Local search to improve the set U."""
    U = sorted(set(U))
    best_score = _compute_score(U)
    best_U = U.copy()
    
    for _ in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        improved = False
        max_u = best_U[-1]
        
        # Try adding an element
        for new_elem in range(1, max_u + 20):
            if new_elem in best_U:
                continue
            new_U = sorted(best_U + [new_elem])
            s = _compute_score(new_U)
            if s > best_score + 1e-12:
                best_U = new_U
                best_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing an element (keep 0 and at least one other)
        for i in range(1, len(best_U)):
            new_U = best_U[:i] + best_U[i+1:]
            if len(new_U) < 2:
                continue
            s = _compute_score(new_U)
            if s > best_score + 1e-12:
                best_U = new_U
                best_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping an element
        found_swap = False
        for i in range(1, len(best_U)):
            for new_elem in range(1, max_u + 30):
                if new_elem in best_U:
                    continue
                new_U = sorted(best_U[:i] + [new_elem] + best_U[i+1:])
                s = _compute_score(new_U)
                if s > best_score + 1e-12:
                    best_U = new_U
                    best_score = s
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
        
        if not improved:
            break
    
    return best_U, best_score

def _construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Systematic search for size 3
    for a in range(1, 25):
        for b in range(a + 1, 45):
            if time.time() > deadline - 10:
                return best_U
            U = [0, a, b]
            s = _compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Systematic search for size 4
    for a in range(1, 18):
        for b in range(a + 1, 30):
            for c in range(b + 1, 50):
                if time.time() > deadline - 10:
                    return best_U
                U = [0, a, b, c]
                s = _compute_score(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # Local search from best found
    if time.time() < deadline - 20:
        improved_set, improved_score = _local_search(best_U, deadline, max_iter=5000)
        if improved_score > best_score:
            best_U = improved_set
            best_score = improved_score
    
    # Try local search from several promising starting points
    starting_points = [
        [0, 1, 4, 6],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 2, 5, 9],
        [0, 1, 5, 8],
        [0, 1, 4, 7],
        [0, 2, 4, 7],
        [0, 1, 3, 5],
        [0, 1, 3, 8],
        [0, 1, 4, 9],
    ]
    
    for start in starting_points:
        if time.time() > deadline - 10:
            break
        s = _compute_score(start)
        if s > best_score - 0.1:
            improved_set, improved_score = _local_search(start, deadline, max_iter=3000)
            if improved_score > best_score:
                best_U = improved_set
                best_score = improved_score
    
    # Final local search
    if time.time() < deadline - 5:
        improved_set, improved_score = _local_search(best_U, deadline, max_iter=3000)
        if improved_score > best_score:
            best_U = improved_set
            best_score = improved_score
    
    return sorted(best_U)

def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    return _construct()

# EVOLVE-BLOCK-END
