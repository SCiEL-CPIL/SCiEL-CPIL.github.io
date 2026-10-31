import time
import math
import random
import numpy as np
from itertools import combinations

_START_TIME = time.time()
TIME_LIMIT = 115

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return TIME_LIMIT - _elapsed()

def _score_set(U):
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    n = len(U_arr)
    
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    if n > 1:
        i_idx, j_idx = np.triu_indices(n, k=1)
        pos_diffs = U_arr[i_idx] - U_arr[j_idx]
        pos_diff_unique = len(np.unique(pos_diffs))
        diffset_size = 2 * pos_diff_unique + 1
    else:
        diffset_size = 1
    
    if sumset_size == 0:
        return 0.0
    
    log_ratio = math.log(diffset_size / sumset_size)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + log_ratio / log_denom

def _exhaustive_small(max_u, min_size, max_size):
    best_U = [0]
    best_score = _score_set([0])
    
    for size in range(min_size, max_size + 1):
        if _time_remaining() < 5:
            break
        # Generate combinations of 0..max_u of given size, ensuring 0 is included
        # We fix 0 and choose size-1 from 1..max_u
        for combo in combinations(range(1, max_u + 1), size - 1):
            if _time_remaining() < 1:
                break
            U = [0] + list(combo)
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_U = U
    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1, 4, 16]
    best_score = _score_set(best_U)
    
    # Search small sets
    # n=3, M<=20
    U3, S3 = _exhaustive_small(20, 3, 3)
    if S3 > best_score:
        best_score = S3
        best_U = U3
        
    # n=4, M<=30
    U4, S4 = _exhaustive_small(30, 4, 4)
    if S4 > best_score:
        best_score = S4
        best_U = U4
        
    # n=5, M<=50
    if _time_remaining() > 20:
        U5, S5 = _exhaustive_small(50, 5, 5)
        if S5 > best_score:
            best_score = S5
            best_U = U5
            
    # n=6, M<=60
    if _time_remaining() > 10:
        U6, S6 = _exhaustive_small(60, 6, 6)
        if S6 > best_score:
            best_score = S6
            best_U = U6

    return sorted(set(best_U))
